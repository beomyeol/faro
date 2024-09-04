import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.dataprocessing.transformers import Scaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pred.utils import build_dataset_from_df, get_likelihood, get_optim_cls


def main(args):
    pl.seed_everything(args.seed)

    df = pd.read_pickle(args.input_path)
    if args.cluster is not None:
        print(f"Using traces with cluster={args.cluster}", flush=True)
        df = df[df.cluster == args.cluster]
    if args.idx is not None:
        target = sorted(df.hash_func.unique())[args.idx]
        print(f"Using {args.idx}-th trace. {target[:5]}", flush=True)
        df = df[df.hash_func == target]

    has_sample_col = "sample" in df.columns
    dataset = build_dataset_from_df(df, check_validity=(not has_sample_col))
    # dataset_ts is ordered by 'group'
    dataset_ts = TimeSeries.from_group_dataframe(
        dataset, group_cols="group", time_col="time_idx", value_cols="value")
    if args.no_scale or args.likelihood == "negbinomial":
        print("No scale", flush=True)
        scaler = None
        scaled = dataset_ts
    else:
        print("Standard scale", flush=True)
        scaler = Scaler(StandardScaler())
        scaled = scaler.fit_transform(dataset_ts)

    if has_sample_col:
        # regenerate dataset for each sample by using scaler for each function
        scaled = []
        for sample_idx in sorted(df["sample"].unique()):
            dataset = build_dataset_from_df(df[df["sample"] == sample_idx])
            dataset_ts = TimeSeries.from_group_dataframe(
                dataset, group_cols="group", time_col="time_idx", value_cols="value")
            # scaler order will be the same as the dataset_ts order
            if scaler is not None:
                scaled.extend(scaler.transform(dataset_ts))
            else:
                scaled.extend(dataset_ts)

    count_len = int(df.counts.agg(lambda x: x.size).max())
    print(f"max count len: {count_len}", flush=True)

    # splits = [ts_item.split_after(count_len-1) for ts_item in scaled]
    # val, train = zip(*splits)
    # train = [ts.shift(-count_len) for ts in train]
    splits = [ts_item.split_after(len(ts_item) - count_len - 1) for ts_item in scaled]
    train, val = zip(*splits)

    print(f"#train: {len(train)}, #val: {len(val)}", flush=True)
    for i in range(len(train)):
        print(f"\ttrain {i}. length={len(train[i])}", flush=True)
    for i in range(len(val)):
        print(f"\tval {i}. length={len(val[i])}", flush=True)

    model_path = Path(args.log_dir).joinpath("model.pt")

    if model_path.exists():
        print(f"Loading models from {model_path}", flush=True)
        model = NHiTSModel.load(model_path.as_posix())
        # update gpu
        model.trainer_params["devices"] = [args.gpu]
    else:
        early_stopper = EarlyStopping(
            "val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
        callbacks = [early_stopper]

        pl_trainer_kwargs = {"callbacks": callbacks}
        if args.gpu is not None:
            pl_trainer_kwargs["accelerator"] = "gpu"
            pl_trainer_kwargs["devices"] = [args.gpu]
            pl_trainer_kwargs["auto_select_gpus"] = True

        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

        likelihood = None
        if args.likelihood is not None:
            likelihood = get_likelihood(args.likelihood)

        model = NHiTSModel(
            input_chunk_length=args.context_len,
            output_chunk_length=args.pred_len,
            optimizer_cls=get_optim_cls(args.optim),
            optimizer_kwargs={"lr": args.lr},
            n_epochs=args.epochs,
            num_blocks=args.blocks,
            num_layers=args.layers,
            num_stacks=args.stacks,
            log_tensorboard=True,
            save_checkpoints=True,
            pl_trainer_kwargs=pl_trainer_kwargs,
            model_name="model",
            layer_widths=args.layer_width,
            force_reset=False,
            work_dir=args.log_dir,
            dropout=args.dropout,
            batch_size=args.batch_size,
            likelihood=likelihood,
        )
    if args.lr_find:
        train_dataset = model._build_train_dataset(train, None, None, None)
        net = model._create_model(train_dataset[0])
        trainer = model._init_trainer(model.trainer_params)
        train_loader = DataLoader(
            train_dataset,
            batch_size=model.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            collate_fn=model._batch_collate_fn,
            num_workers=args.num_loader_workers,
        )
        lr_finder = trainer.tuner.lr_find(
            net,
            train_dataloaders=train_loader,
            num_training=200,
        )
        # model.trainer_params["max_steps"] = 1
        # model.fit(series=train, val_series=val, verbose=args.verbose,
        #           num_loader_workers=args.num_loader_workers)
        # model.trainer_params["max_steps"] = -1
        # model._setup_trainer(verbose=args.verbose)
        # lr_finder = model.model.trainer.tuner.lr_find(
        #     model.model,
        #     train_dataloaders=model.model.trainer._data_connector._train_dataloader_source.instance,
        #     val_dataloaders=model.model.trainer._data_connector._val_dataloader_source.instance,
        # )
        print(f"{lr_finder.suggestion()}", flush=True)
    else:
        model.fit(series=train, val_series=val, verbose=args.verbose,
                  num_loader_workers=args.num_loader_workers)

        model.save(Path(args.log_dir).joinpath("model.pt").as_posix())
        torch.save(scaler, Path(args.log_dir).joinpath("scaler.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--cluster", type=int)

    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optim", type=str, default="adam")

    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--stacks", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--layer-width", type=int, default=512)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--likelihood", type=str)
    parser.add_argument("--idx", type=int)
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--num-loader-workers", type=int, default=4)
    parser.add_argument("--lr-find", action="store_true")

    args = parser.parse_args()

    print(args, flush=True)

    main(args)
