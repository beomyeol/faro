import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir")
    parser.add_argument("-d", "--day", type=int, required=True)
    parser.add_argument("-b", "--base", type=int, default=0)
    parser.add_argument("-c", "--count", type=int, default=5)
    parser.add_argument("--targets", help="comma seperated target functions")
    parser.add_argument("--min-qps", type=int, help="change min qps")
    parser.add_argument("--max-qps", type=int, help="change max qps")
    parser.add_argument("-o", "--out", required=True, help="output path")

    args = parser.parse_args()
    print(args)

    invocation_fpattern = "invocations_per_function_md.anon.d%02d.csv"

    fpath = Path(args.inputdir).joinpath(
        invocation_fpattern % args.day).resolve()
    print(f"Opening {fpath}")
    df = pd.read_csv(fpath)

    print(f"---- Day {args.day} ----")
    print("# unique app:", len(df.HashApp.unique()))
    print("# unique func:", len(df.HashFunction.unique()))
    print("# entries:", len(df))
    sums = np.array([df[f"{x}"].sum() for x in np.arange(1, 1441)])
    print("total # invocations:", np.sum(sums))
    print("max qps:", np.max(sums))
    print("-----------------")

    count_df = df[[str(i) for i in np.arange(1, 1441)]]
    count_sum_df = count_df.sum(axis=1).sort_values(ascending=False)

    if args.targets:
        targets = [int(target) for target in args.targets.split(",")]
    else:
        targets = count_sum_df[args.base:args.base+args.count].index.to_numpy()

    print(f"targets: {targets}")

    samples = df.iloc[targets]
    keys = []
    counts = []
    for index, s in samples.iterrows():
        keys.append(f"f_{index}")
        counts.append(np.array([s[str(i)] for i in np.arange(1, 1441)]))
    counts = np.stack(counts)

    print(f"---- Samples stats ----")

    def print_stats(keys, counts):
        for k, v in zip(keys, counts):
            print(
                f"{k}, avg={np.mean(v):.2f}, max={np.max(v)}, min={np.min(v)}, sum={np.sum(v)}")
        total_count = np.sum(counts, axis=0)
        print(f"min qps: {np.min(total_count)}")
        print(f"max qps: {np.max(total_count)}")
        print(f"total counts: {np.sum(total_count)}")

    print_stats(keys, counts)

    if args.min_qps or args.max_qps:
        total_count = np.sum(counts, axis=0)
        min_qps = np.min(total_count)
        max_qps = np.max(total_count)
        new_min_qps = args.min_qps or min_qps
        new_max_qps = args.max_qps or max_qps

        counts = np.round(
            (counts - min_qps)/(max_qps - min_qps)
            * (new_max_qps - new_min_qps) + new_min_qps).astype(np.int32)

        print(f"---- Remapped samples stats ----")
        print_stats(keys, counts)

    print(f"Outputing results at {args.out}...")
    with open(args.out, "w") as f:
        for ts in np.arange(counts.shape[1]):
            urls = {k: v.item() for k, v in zip(keys, counts[:, ts])}
            f.write(json.dumps({"ts": int(ts), "urls": urls}))
            f.write("\n")


if __name__ == "__main__":
    main()
