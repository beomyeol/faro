import argparse
from pathlib import Path
import subprocess
import sys
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--target_latency", type=float)
    parser.add_argument("--out_dir")

    args = parser.parse_args()

    dir_path = Path(args.dir).resolve()
    config_path = dir_path.joinpath("config.yaml")

    root_dir = Path(__file__).resolve().parents[2]
    src_dir = root_dir.joinpath("src")

    if args.out_dir is not None:
        out_dir_path = Path(args.out_dir)
        out_dir_path.mkdir(exist_ok=True, parents=True)
        sys.path.append(src_dir.resolve().as_posix())
        from configuration import load_config
        config = load_config(config_path)
        with open(out_dir_path.joinpath("config.yaml"), "w") as f:
            yaml.safe_dump(config, f)
    else:
        out_dir_path = dir_path

    cmd_dir = [
        "python", "-m", "simulation.simulation", str(config_path),
        # f"--metric_out_dir={out_dir_path.as_posix()}"
        f"--metric_dump_path={out_dir_path.joinpath('metrics.pkl.gz').as_posix()}"
    ]
    if args.target_latency is not None:
        cmd_dir.append(f"--target_latency={args.target_latency}")

    p = subprocess.Popen(cmd_dir, cwd=src_dir, stderr=subprocess.STDOUT,
                         stdout=subprocess.PIPE)
    with open(out_dir_path.joinpath("sim_run.log"), "w") as f:
        for line in iter(p.stdout.readline, b''):
            text = line.decode('utf-8')
            f.write(text)
            print(text, end="")
    p.wait()

    sys.exit(p.returncode)
