import argparse
import os
import subprocess
import tempfile
from pathlib import Path


def build_model_cmd(args):
    model_cmd = []

    input_path = Path(args.input_path).resolve()

    input_name = os.path.splitext(input_path.name)[0]
    if args.cluster is not None:
        input_name += f"_c={args.cluster}"
        model_cmd.append(f"--cluster={args.cluster}")
    if args.idx is not None:
        input_name += f"_i={args.idx}"
        model_cmd.append(f"--idx={args.idx}")

    log_dir = root_dir.joinpath(args.out_dir).joinpath("pred").\
        joinpath(input_name).\
        joinpath(f"seed={args.seed}").\
        joinpath(args.tool).\
        joinpath(f"clen={args.context_len}_plen={args.pred_len}").\
        joinpath(args.model_tag if args.model_tag is not None else args.model_name).\
        joinpath(f"bs={args.batch_size}_lr={args.lr}")
    if args.model_name == "nhits":
        log_dir = log_dir.joinpath(
            f"b={args.blocks}_s={args.stacks}_l={args.layers}_lw={args.layer_width}_do={args.dropout}")
        model_cmd += [
            f"--blocks={args.blocks}",
            f"--layers={args.layers}",
            f"--stacks={args.stacks}",
            f"--layer-width={args.layer_width}",
            f"--dropout={args.dropout}",
        ]
        if args.likelihood is not None:
            log_dir = log_dir.with_name(log_dir.name + "_" + args.likelihood)
            model_cmd.append(f"--likelihood={args.likelihood}")
        if args.loss is not None:
            log_dir = log_dir.with_name(log_dir.name + "_" + args.loss)
            model_cmd.append(f"--loss={args.loss}")
    elif args.model_name == "tft":
        log_dir = log_dir.joinpath(
            f"l={args.layers}_lw={args.layer_width}_ah={args.num_attention_heads}_do={args.dropout}")
        model_cmd += [
            f"--layers={args.layers}",
            f"--layer-width={args.layer_width}",
            f"--num-attention-heads={args.num_attention_heads}",
            f"--dropout={args.dropout}",
        ]
        if args.likelihood is not None:
            log_dir = log_dir.with_name(log_dir.name + "_" + args.likelihood)
            model_cmd.append(f"--likelihood={args.likelihood}")
        if args.full_attention:
            log_dir = log_dir.with_name(log_dir.name + "_fa")
            model_cmd.append("--full-attention")
    elif args.model_name == "rnn" or args.model_name == "deepar":
        log_dir = log_dir.joinpath(args.type).\
            joinpath(
                f"l={args.layers}_lw={args.layer_width}_do={args.dropout}")
        model_cmd += [
            f"--type={args.type}",
            f"--layers={args.layers}",
            f"--layer-width={args.layer_width}",
            f"--dropout={args.dropout}",
        ]
        if args.likelihood is not None:
            log_dir = log_dir.with_name(log_dir.name + "_" + args.likelihood)
            model_cmd.append(f"--likelihood={args.likelihood}")
        if args.loss is not None:
            log_dir = log_dir.with_name(log_dir.name + "_" + args.loss)
            model_cmd.append(f"--loss={args.loss}")
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    if args.no_scale:
        log_dir = log_dir.with_name(log_dir.name + "_noscale")
        model_cmd.append("--no-scale")

    return log_dir, model_cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--cluster", type=int)
    parser.add_argument("--tool", type=str,
                        choices=["ptf", "darts", "gluonts"], required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--pred-len", type=int, required=True)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-tag", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--idx", type=int)
    parser.add_argument("--no-scale", action="store_true")

    # nhits & tft
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--likelihood", type=str)
    # rnn
    parser.add_argument(
        "--type", choices=["RNN", "LSTM", "GRU"], default="LSTM")
    # rnn & nhits
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--stacks", type=int, default=3)
    # rnn & nhits & tft
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--layer-width", type=int, default=512)
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--loss", type=str)
    # tft
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--full-attention", action="store_true")
    # TODO: add more tft flags if needed.

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent

    log_dir, model_cmd = build_model_cmd(args)

    input_path = Path(args.input_path).resolve()

    base_cmd = [
        "python", "-m", f"pred.{args.tool}_{args.model_name}",
        f"{input_path}",
        f"--context-len={args.context_len}",
        f"--pred-len={args.pred_len}",
        f"--batch-size={args.batch_size}",
        f"--epochs={args.epochs}",
        f"--model-name={args.model_name}",
        f"--seed={args.seed}",
    ]

    if args.gpu is not None:
        base_cmd += [f"--gpu={args.gpu}"]

    if args.tool == "darts" and args.lr is None:
        # run lr find
        with tempfile.TemporaryDirectory() as tmpdir_name:
            cmd = base_cmd + model_cmd + [
                f"--lr-find",
                f"--log-dir={tmpdir_name}",
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=root_dir,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
            )
            for line in iter(proc.stdout.readline, b''):
                text = line.decode('utf-8')
                print(text, end="", flush=True)
            proc.wait()
            args.lr = float(f"{float(text):0.1e}")

    log_dir, model_cmd = build_model_cmd(args)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = base_cmd + model_cmd + [
        f"--lr={args.lr}",
        f"--log-dir={log_dir}",
    ]

    print(" ".join(cmd), flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=root_dir,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
    )
    with open(log_dir.joinpath("run.log"), "w") as f:
        for line in iter(proc.stdout.readline, b''):
            text = line.decode('utf-8')
            f.write(text)
            f.flush()
            print(text, end="", flush=True)
    proc.wait()
