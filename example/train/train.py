import argparse
import os
import time
from filelock import FileLock

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import horovod.torch as hvd

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    metavar='N',
    help='input batch size for training (default: 64)')
parser.add_argument(
    '--test-batch-size',
    type=int,
    default=1000,
    metavar='N',
    help='input batch size for testing (default: 1000)')
parser.add_argument(
    '--epochs',
    type=int,
    default=10,
    metavar='N',
    help='number of epochs to train (default: 10)')
parser.add_argument(
    '--lr',
    type=float,
    default=0.01,
    metavar='LR',
    help='learning rate (default: 0.01)')
parser.add_argument(
    '--momentum',
    type=float,
    default=0.5,
    metavar='M',
    help='SGD momentum (default: 0.5)')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--seed',
    type=int,
    default=42,
    metavar='S',
    help='random seed (default: 42)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='how many batches to wait before logging training status')
parser.add_argument(
    '--fp16-allreduce',
    action='store_true',
    default=False,
    help='use fp16 compression during allreduce')
parser.add_argument(
    '--use-adasum',
    action='store_true',
    default=False,
    help='use adasum algorithm to do reduction')
parser.add_argument(
    '--num-batches-per-commit',
    type=int,
    default=1,
    help='number of batches per commit of the elastic state object'
)
parser.add_argument(
    '--data-dir',
    help='location of the training dataset in the local filesystem (will be downloaded if needed)'
)
parser.add_argument(
    '--min-workers',
    type=int,
    default=1,
    help='min ray workers'
)

args = parser.parse_args()


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train_fn():
    # Horovod: initialize library.
    hvd.init()
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    data_dir = args.data_dir or './data'
    with FileLock(os.path.expanduser("~/.horovod_lock")):
        train_dataset = \
            datasets.MNIST(data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
    train_sampler = hvd.elastic.ElasticSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        **kwargs)
    transformations = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    test_dataset = datasets.MNIST(
        data_dir, train=False, transform=transformations)
    test_sampler = hvd.elastic.ElasticSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        **kwargs)

    model = Net()

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size() if not args.use_adasum else 1

    if args.cuda:
        # Move model to GPU.
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    # Horovod: scale learning rate by lr_scaler.
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr * lr_scaler, momentum=args.momentum)

    # Horovod: (optional) compression algorithm.
    compression = (hvd.Compression.fp16
                   if args.fp16_allreduce else hvd.Compression.none)

    def train(state):
        # post synchronization event (worker added, worker removed) init ...
        state.model.train()
        epoch = state.epoch
        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        batch_offset = state.batch

        for idx, (data, target) in enumerate(train_loader):
            state.batch = batch_offset + idx
            if state.batch % args.num_batches_per_commit == 0:
                state.commit()
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            state.optimizer.zero_grad()
            output = state.model(data)
            loss = F.nll_loss(output, target)
            train_loss.update(loss)
            loss.backward()
            state.train_sampler.record_batch(idx, args.batch_size)
            state.optimizer.step()
            if state.batch % args.log_interval == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.
                      format(state.epoch, idx, len(train_loader),
                             loss.item()))
        state.commit()

    def test(epoch):
        model.eval()
        test_loss = Metric('val_loss')
        test_accuracy = Metric('val_accuracy')
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                # sum up batch loss
                test_loss.update(F.nll_loss(output, target, size_average=False))
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
                test_accuracy.update(pred.eq(
                    target.data.view_as(pred)).cpu().float().mean())

        # Horovod: print output only on first rank.
        if hvd.rank() == 0:
            print(
                '\nTest Epoch: {} Average loss: {:.4f}, Accuracy: {:.2f}%\n'.
                format(epoch, test_loss.avg, 100. * test_accuracy.avg))

    @hvd.elastic.run
    def full_train(state):
        while state.epoch < args.epochs:
            ts = time.time()
            train(state)
            # test(state.epoch)
            state.epoch += 1
            state.batch = 0
            state.train_sampler.set_epoch(state.epoch)
            state.commit()
            print(f"Epoch: {state.epoch -1} Elapsed time: {time.time() - ts} Rank: {hvd.rank()}", flush=True)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        op=hvd.Adasum if args.use_adasum else hvd.Average)

    # adjust learning rate on reset
    def on_state_reset():
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * hvd.size()

    state = hvd.elastic.TorchState(
        model, optimizer, train_sampler=train_sampler,
        test_sampler=test_sampler, epoch=0, batch=0)
    state.register_reset_callbacks([on_state_reset])
    full_train(state)


if __name__ == '__main__':
    from horovod.ray import RayExecutor
    import ray
    ray.init(address="auto")
    settings = RayExecutor.create_settings(timeout_s=60)
    executor = RayExecutor(settings, min_workers=args.min_workers,
                           use_gpu=False, cpus_per_worker=1)
    executor.start()
    executor.run(train_fn)
