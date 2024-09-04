import argparse
import pandas as pd

import pytorch_lightning as pl
from pytorch_forecasting import DeepAR, TimeSeriesDataSet, EncoderNormalizer
from pytorch_forecasting.metrics import NormalDistributionLoss, MQF2DistributionLoss, RMSE, NegativeBinomialDistributionLoss
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from pred.utils import build_dataset_from_df


def main(args):
    pl.seed_everything(args.seed)

    df = pd.read_pickle(args.input_path)
    if args.idx is not None:
        target = sorted(df.hash_func.unique())[args.idx]
        print(f"Using {args.idx}-th trace. {target[:5]}", flush=True)
        df = df[df.hash_func == target]

    train = build_dataset_from_df(df[df.day != 8])
    val = build_dataset_from_df(df[df.day == 8])

    if args.loss is not None and args.loss == "negbinomial":
        target_normalizer = EncoderNormalizer(center=False)
    else:
        target_normalizer = "auto"

    train = TimeSeriesDataSet(
        train,
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        time_varying_unknown_reals=["value"],
        max_encoder_length=args.context_len,
        max_prediction_length=args.pred_len,
        target_normalizer=target_normalizer,
        # target_normalizer=EncoderNormalizer(),
        # target_normalizer=EncoderNormalizer(transformation="relu"),
    )
    print("train_param:", train.get_parameters(), flush=True)

    val = TimeSeriesDataSet.from_dataset(train, val)
    print("val_param:", val.get_parameters(), flush=True)

    batch_size = args.batch_size
    num_sequences = len(train.index.sequence_id.unique())
    if batch_size > num_sequences:
        raise ValueError(
            f"batch_size ({batch_size}) > num_sequences ({num_sequences})")

    # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
    train_dataloader = train.to_dataloader(
        train=True, batch_size=batch_size,  # batch_sampler="synchronized"
    )
    val_dataloader = val.to_dataloader(
        train=False, batch_size=batch_size,  # batch_sampler="synchronized"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")

    logger = pl_loggers.TensorBoardLogger(save_dir=args.log_dir)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        devices=(None if args.gpu is None else [args.gpu]),
        accelerator=(None if args.gpu is None else "gpu"),
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback],
        # limit_train_batches=50,
        enable_checkpointing=True,
        enable_progress_bar=args.verbose,
    )

    loss = None
    if args.loss is not None:
        if args.loss == "mse":
            loss = RMSE()
        elif args.loss == "mqf2":
            loss = MQF2DistributionLoss(prediction_length=args.pred_len)
        elif args.loss == "negbinomial":
            loss = NegativeBinomialDistributionLoss()
        else:
            raise ValueError(f"Unsupported loss: {args.loss}")

    net = DeepAR.from_dataset(
        train,
        cell_type=args.type,
        learning_rate=args.lr,
        log_interval=200,
        log_val_interval=200,
        hidden_size=args.layer_width,
        rnn_layers=args.layers,
        dropout=args.dropout,
        n_plotting_samples=0,
        # weight_decay=1e-3,
        loss=loss,
    )

    res = trainer.tuner.lr_find(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        num_training=500,
        min_lr=1e-5,
        max_lr=1e-3,
        # early_stop_threshold=100,
    )
    print(f"suggested learning rate: {res.suggestion()}", flush=True)
    net.hparams.learning_rate = res.suggestion()

    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(best_model_path, flush=True)


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--type", choices=["LSTM", "GRU"], default="LSTM")

    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--layer-width", type=int, default=512)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--loss", type=str)
    parser.add_argument("--idx", type=int)

    args = parser.parse_args()

    print(args, flush=True)

    main(args)
