import argparse
from pathlib import Path
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from gluonts.dataset.split import split
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator

from pred.utils import build_dataset_from_df, get_distr_output


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

    start = '2023-01-01'
    freq = 'min'  # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    if "sample" in df.columns:
        dataset_list = []
        for sample_idx in sorted(df["sample"].unique()):
            dataset = build_dataset_from_df(df[df["sample"] == sample_idx])
            for group in dataset.group.unique():
                dataset_list.append({
                    'target': dataset[dataset.group == group].value.to_numpy(),
                    'start': start
                })
        data = ListDataset(dataset_list, freq=freq)
    else:
        dataset = build_dataset_from_df(df)
        data = ListDataset(
            [{'target': dataset[dataset.group == group].value.to_numpy(),
              'start': start} for group in dataset.group.unique()],
            freq=freq)

    count_len = int(df.counts.agg(lambda x: x.size).max())
    print(f"max count len: {count_len}", flush=True)

    distance = 2
    max_test_samples = count_len - args.context_len + 1
    windows = int(max_test_samples // distance)

    train, test_gen = split(data, offset=-count_len)
    test = test_gen.generate_instances(
        prediction_length=args.pred_len,
        windows=windows,
        distance=distance,
    )
    print(f"#train: {len(train)}, #val: {len(test)}", flush=True)

    early_stopper = EarlyStopping(
        "val_loss", min_delta=1e-5, patience=20, verbose=False, mode="min")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(args.log_dir).joinpath("checkpoints").as_posix(),
        save_last=True,
        monitor="val_loss",
        filename="best-{epoch}-{val_loss:.2f}",
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "last-{epoch}"
    callbacks = [early_stopper, checkpoint_callback]

    pl_trainer_kwargs = {
        "callbacks": callbacks,
        "max_epochs": args.epochs,
        "logger": pl_loggers.TensorBoardLogger(
            save_dir=Path(args.log_dir).joinpath("logs").as_posix(),
            name="",
        )
    }
    if args.gpu is not None:
        pl_trainer_kwargs["accelerator"] = "gpu"
        pl_trainer_kwargs["devices"] = [args.gpu]
        pl_trainer_kwargs["auto_select_gpus"] = True
        pl_trainer_kwargs["enable_progress_bar"] = args.verbose

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=args.pred_len,
        context_length=args.context_len,
        num_layers=args.layers,
        hidden_size=args.layer_width,
        lr=args.lr,
        dropout_rate=args.dropout,
        batch_size=args.batch_size,
        trainer_kwargs=pl_trainer_kwargs,
        scaling=(not args.no_scale),
        num_batches_per_epoch=100,
        distr_output=get_distr_output(args.likelihood),
        lags_seq=[0]
    )

    predictor = estimator.train(train, test.input, num_workers=4)

    predictor.serialize(Path(args.log_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--cluster", type=int)

    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--type", choices=["LSTM"], default="LSTM")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--layer-width", type=int, default=40)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--idx", type=int)
    parser.add_argument("--likelihood", type=str, default="student")
    parser.add_argument("--no-scale", action="store_true")

    args = parser.parse_args()

    print(args, flush=True)

    main(args)
