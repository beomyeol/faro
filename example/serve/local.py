import torch
import torchvision
import torchvision.transforms as transforms
import requests
from PIL import Image
import io
import time
import numpy
import argparse
import tqdm
import pickle


def run_with_real_image(model, preprocessor, args):
    img_bytes = requests.get(
        "https://cdn.britannica.com/79/65379-050-5CF52BAC/Shortfin-mako-shark-seas.jpg").content

    print(len(img_bytes))

    latencies = []
    for _ in tqdm.tqdm(range(args.repeat)):
        start_ts = time.time()
        input_tensor = preprocessor(
            Image.open(io.BytesIO(img_bytes))).unsqueeze(0)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        class_index = int(torch.argmax(output_tensor[0]))
        latency = time.time() - start_ts
        latencies.append(latency)

    return latencies


def run_with_dataset(model, preprocessor, args):
    if args.dataset == "flower102":
        dataset = torchvision.datasets.Flowers102(
            root=args.dataset_root, download=True)
    else:
        raise ValueError(f"unknown dataset: {args.dataset}")

    latencies = []
    for img, _ in tqdm.tqdm(dataset):
        start_ts = time.time()
        input_tensor = preprocessor(img).unsqueeze(0)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        class_index = int(torch.argmax(output_tensor[0]))
        latency = time.time() - start_ts
        latencies.append(latency)
    return latencies


def main(args):
    device = torch.device("cpu")
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("--gpu is used but cuda is not available")
    model_func = getattr(torchvision.models, args.model)
    model = model_func(pretrained=True).to(device).eval()

    preprocessor = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # remove alpha channel and move tensor device
        transforms.Lambda(lambda t: t[:3, ...].to(device)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # warm up
    for img, _ in torchvision.datasets.FakeData(size=10):
        with torch.no_grad():
            model(preprocessor(img).unsqueeze(0))

    if args.dataset is not None:
        latencies = run_with_dataset(model, preprocessor, args)
    else:
        latencies = run_with_real_image(model, preprocessor, args)

    if args.out is not None:
        print(f"writing results to {args.out}...")
        with open(args.out, "wb") as f:
            pickle.dump(latencies, f)

    print(f"average latency: {numpy.mean(latencies)}")
    print(f"min latency: {numpy.min(latencies)}")
    print(f"median latency: {numpy.percentile(latencies, 50)}")
    print(f"90 percentile: {numpy.percentile(latencies, 90)}")
    print(f"max latency: {numpy.max(latencies)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model to use")
    parser.add_argument(
        "--dataset", choices=["flower102"], help="dataset to use")
    parser.add_argument("--dataset_root", default=".", help="dataset root dir")
    parser.add_argument("--repeat", type=int, default=100, help="num repeats")
    parser.add_argument("--gpu", action="store_true",
                        help="use gpu if available")
    parser.add_argument("--out",
                        help="output path to store pickled latency results")
    main(parser.parse_args())
