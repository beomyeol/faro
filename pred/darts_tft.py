import argparse
from pathlib import Path

import pandas as pd
import torch
from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pred.utils import build_dataset_from_df, get_likelihood, get_optim_cls


def main(args):
    pl.seed_everything(args.seed)

    df = pd.read_pickle(args.input_path)
    train = build_dataset_from_df(df[df.day != 8])
    val = build_dataset_from_df(df[df.day == 8])

    train = TimeSeries.from_group_dataframe(
        train, group_cols="group", time_col="time_idx", value_cols="value")
    val = TimeSeries.from_group_dataframe(
        val, group_cols="group", time_col="time_idx", value_cols="value")
    print(f"#train: {len(train)}, #val: {len(val)}", flush=True)

    scaler = Scaler(StandardScaler())
    scaler.fit(train + val)
    scaled_train = scaler.transform(train)
    scaled_val = scaler.transform(val)

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

    model = TFTModel(
        input_chunk_length=args.context_len,
        output_chunk_length=args.pred_len,
        optimizer_cls=get_optim_cls(args.optim),
        optimizer_kwargs={"lr": args.lr},
        n_epochs=args.epochs,
        lstm_layers=args.layers,
        log_tensorboard=True,
        save_checkpoints=True,
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="model",
        hidden_size=args.layer_width,
        force_reset=True,
        work_dir=args.log_dir,
        dropout=args.dropout,
        batch_size=args.batch_size,
        likelihood=likelihood,
        num_attention_heads=args.num_attention_heads,
        full_attention=args.full_attention,
    )
    model.fit(series=scaled_train, val_series=scaled_val, verbose=args.verbose)

    model.save(Path(args.log_dir).joinpath("model.pt").as_posix())
    torch.save(scaler, Path(args.log_dir).joinpath("scaler.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--optim", type=str, default="adam")

    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--layer-width", type=int, default=128)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--full-attention", action="store_true")

    parser.add_argument("--model-name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--likelihood", type=str)

    args = parser.parse_args()

    print(args, flush=True)

    main(args)
