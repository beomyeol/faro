
import argparse
from pathlib import Path
import pickle
import time

import pandas as pd
import numpy as np
from sktime.clustering.k_means import TimeSeriesKMeans
# from sktime.clustering.k_shapes import TimeSeriesKShapes  # doesn't work
from tslearn.clustering import KShape
from sklearn.preprocessing import StandardScaler


def get_concat_df(df):
    concat_dict = {}
    for v, s in df.sort_values(["hash_func", "day"]).groupby("hash_func"):
        concat_dict[v] = pd.concat(s.counts.values, ignore_index=True)
    return pd.DataFrame(
        {"hash_func": concat_dict.keys(), "counts": concat_dict.values()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--metric")
    parser.add_argument("--out-dir", default="cluster_log")
    parser.add_argument("--day", type=int)
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--algo", default="kmeans",
                        choices=["kmeans", "kshapes"])
    parser.add_argument("--init")
    parser.add_argument("--concat", action="store_true")
    parser.add_argument("--aggr-window-size", type=int)
    parser.add_argument("--aggr", choices=["max", "avg"])

    args = parser.parse_args()
    print(args, flush=True)

    input_path = Path(args.input_path)

    df: pd.DataFrame = pd.read_pickle(input_path)

    if args.day is not None:
        target_df = df[df.day == args.day]
    else:
        target_df = df

    if args.concat:
        print("Using concatenated series for each function", flush=True)
        target_df = get_concat_df(target_df)

    counts = np.stack([c.to_numpy() for c in target_df.counts], axis=-1)
    if args.aggr is not None:
        assert args.aggr_window_size is not None
        if args.aggr == "max":
            aggr_fn = np.max
        elif args.aggr == "avg":
            aggr_fn = np.mean
        else:
            raise ValueError("Unknown aggr: " + args.aggr)
        split = np.split(
            counts,
            np.arange(args.aggr_window_size, len(counts), args.aggr_window_size))
        counts = aggr_fn(split, axis=1)

    if args.scale:
        scaler = StandardScaler()
        counts = scaler.fit_transform(counts)
    counts = counts.transpose(1, 0)
    print(counts.shape, flush=True)
    print(f"mean: {np.mean(counts, axis=1)}")
    print(f"stdev: {np.std(counts, axis=1)}")

    kwargs = {
        "n_clusters": args.clusters,
    }
    if args.init is not None:
        kwargs["init_algorithm"] = args.init
    if args.metric is not None:
        kwargs["metric"] = args.metric

    if args.algo == "kmeans":
        clusterer = TimeSeriesKMeans(**kwargs)
        metric = kwargs.get("metric", "dtw")
        init = kwargs.get("init_algorithm", "forgy")
        out_name = f"kmeans_{metric}_{args.clusters}_{init}"
    elif args.algo == "kshapes":
        # clusterer = TimeSeriesKShapes(**kwargs, verbose=True)
        clusterer = KShape(**kwargs, verbose=True, n_init=10)
        out_name = f"kshapes_{args.clusters}"
    else:
        raise ValueError("unknown algo:", args.algo)

    start_ts = time.time()
    clusterer.fit(counts)
    print(f"Elapsed time (s): {time.time() - start_ts}", flush=True)

    if args.day is not None:
        out_name += f"_day{args.day}"
    if args.concat:
        out_name += "_concat"
    if args.aggr is not None:
        out_name += f"_{args.aggr}_{args.aggr_window_size}"

    out_dir = Path(args.out_dir).joinpath(out_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_out_path = out_dir.joinpath("model.pkl")
    print(f"Outputting model: {str(model_out_path)}")
    with open(model_out_path, "wb") as f:
        pickle.dump(clusterer, f)

    pred = clusterer.predict(counts)

    df["cluster"] = np.zeros(len(df), dtype=np.int32)
    if args.day is not None or args.concat:
        for f, p in zip(target_df.hash_func, pred):
            df.loc[df.hash_func == f, "cluster"] = int(p)
    else:
        df["cluster"] = pred

    df_out_path = out_dir.joinpath(f"{input_path.name}")
    print(f"Outputting dataframe: {str(df_out_path)}")
    df.to_pickle(df_out_path.as_posix())
