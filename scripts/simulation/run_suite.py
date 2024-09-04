import argparse
from collections import defaultdict
from copy import deepcopy
import itertools
import os
from pathlib import Path
import pickle
import random
from statistics import mean
import subprocess
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
import time
from typing import Optional
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.utils import calculate_aggregate
from src.configuration import load_config
from src.utility import create_utility_func, SLOPenalty
from data import utils


def get_num_deployed_replicas(target_dir, max_ts):
    result = defaultdict(list)
    pattern = "#replicas: "
    for line in Path(target_dir).joinpath("sim_run.log").read_text().splitlines():
        if pattern in line:
            tss = utils.parse_operator_ts(line, 0)
            if tss > max_ts:
                break
            splits = line.split(" ")
            cluster_name, deployment_name = splits[3][1:-1].split(":")
            num = int(splits[5])
            result[(cluster_name, deployment_name)].append((tss, num))
    return result


def calculate_total_replica_changes(num_replicas_deployed_dict):
    d = defaultdict(list)
    for i, cs in enumerate(num_replicas_deployed_dict.values()):
        for ts, n in cs:
            d[ts].append((i, n))
    num_replica_changes = {}
    current_num_replicas = np.zeros(len(num_replicas_deployed_dict))
    for ts, vs in sorted(d.items()):
        for i, n in vs:
            current_num_replicas[i] = n
        num_replica_changes[ts] = deepcopy(current_num_replicas)
    return num_replica_changes


def avg_used_replicas(num_replica_changes):
    current_ts = 0
    prev_num_replicas = 0
    used_num_replicas_sum = 0.
    for ts, num_replicas in sorted(num_replica_changes.items()):
        window_size = ts - current_ts
        used_num_replicas_sum += window_size * prev_num_replicas
        current_ts, prev_num_replicas = ts, num_replicas.item()
    return used_num_replicas_sum/current_ts


def draw_replica_changes(out_path, num_replicas_deployed_dict, total_replica_changes, markers):
    # draw replica change fig
    fig, ax = plt.subplots(figsize=(10, 5))
    for (cluster_name, _), num_replicas_deployed in num_replicas_deployed_dict.items():
        ax.step(*zip(*num_replicas_deployed),
                label=f"deployed ({cluster_name})", where="post", marker=next(markers))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("num replicas")

    ax.step(*zip(*total_replica_changes.items()), label="total", where="post")
    ax.set_title("# replica changes")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def get_target_slo(config):
    if "controller" in config:
        if config["controller"] == "autoscale":
            autoscale_config = config["controller_params"]["autoscale_config"]
        elif config["controller"] == "hybrid":
            autoscale_config = config["controller_params"]["policies"][0]["autoscale_config"]
        elif config["controller"] == "default":
            return 720
        else:
            raise ValueError(f"Unsupported controller: {config['controller']}")
    else:
        if config["policy_name"] == "hybrid":
            autoscale_config = config["policies"][0]["autoscale_config"]
        else:
            autoscale_config = config["autoscale_config"]

    cluster_configs = autoscale_config.get("cluster_configs", None)
    if cluster_configs is None:
        target_slo = autoscale_config["target_metric"]
    else:
        target_slo = {
            head_name.replace("-ray-head", ""): cluster_config["target_metric"]
            for head_name, cluster_config in cluster_configs.items()}

    return target_slo


def analyze_result(target_dir: Path, force: bool, utility: str,
                   partial_end: Optional[int], unit_time: Optional[int]):
    print(f"Analyzing {target_dir}. force={force}, utiltiy={utility}, "
          f"partial_end={partial_end}", flush=True)

    if utility is None:
        stats_path = target_dir / "stats.pkl"
    else:
        stats_path = target_dir / f"{utility}_stats.pkl"

    if partial_end is not None:
        stats_path = stats_path.with_name(f"{partial_end}_{stats_path.name}")

    if not force and stats_path.exists():
        stats = pd.read_pickle(stats_path)
        if (utility is None) or (stats.get("utility", None) == utility):
            print(
                f"Done analyzing {target_dir}, load existing stats.", flush=True)
            return stats, target_dir

    analysis_start_ts = time.time()
    config_path = target_dir.joinpath("config.yaml")
    config = load_config(config_path)
    # input_path = Path("src").joinpath(
    #     Path(config["load_specs"][0]["params"]["path"]))
    # unit_time = config["load_specs"][0]["params"].get("unit_time", 1)
    # count_dict = defaultdict(list)
    # for line in input_path.read_text().splitlines():
    #     for cluster, deps in json.loads(line)["counts"].items():
    #         count_dict[cluster].append(sum(deps.values()))

    # if partial_end is not None:
    #     for cluster_name in count_dict.keys():
    #         count_dict[cluster_name] = count_dict[cluster_name][:partial_end]
    #     count_sum_dict = {
    #         cluster_name: np.sum(counts)
    #         for cluster_name, counts in count_dict.items()
    #     }

    metrics_path = target_dir.joinpath("metrics.pkl.gz")
    if metrics_path.exists():
        df = pd.read_pickle(metrics_path)
        cluster_names = df.cluster_name.unique()
        latency_dict = {}
        end_tss = []
        for cluster_name in cluster_names:
            cluster_df = df[df.cluster_name == cluster_name]
            if partial_end is not None:
                # cluster_df = cluster_df.sort_values(by="arrival_ts")[:count_sum_dict[cluster_name]]
                raise NotImplementedError()
            tss = cluster_df.arrival_ts.values
            lats = cluster_df.end_ts - cluster_df.arrival_ts
            lats[~cluster_df.succeeded] = np.nan
            latency_dict[cluster_name] = (tss, lats.values * 1e3)
            end_tss.append(cluster_df.end_ts.max())
        max_ts = np.max(end_tss)

        _, latencies = zip(*latency_dict.values())
        latencies = np.concatenate(latencies)
    else:
        latency_dict = {}
        for out_path in target_dir.glob("time_*.log"):
            cluster_name = out_path.name[len("time_"):out_path.name.rfind(".")]
            tss, lats = [], []
            for line in out_path.read_text().splitlines():
                start_ts, latency = line.split(" ")
                tss.append(float(start_ts))
                lats.append(float(latency))
            latency_dict[cluster_name] = (tss, lats)

        start_times = []
        for tss, _ in latency_dict.values():
            start_times += tss
        latencies = []
        for _, lats in latency_dict.values():
            latencies += lats
        start_times = np.array(start_times)
        latencies = np.array(latencies)

        end_times = start_times + latencies/1e3
        if len(end_times) == 0:
            raise RuntimeError(f"no output at {target_dir}")
        max_ts = max(end_times)

    target_slo = get_target_slo(config)

    if isinstance(target_slo, dict):
        satisfied_count = 0
        for cluster_name, (_, lats) in latency_dict.items():
            satisfied_count += np.count_nonzero(
                np.array(lats) < target_slo[cluster_name])
    else:
        satisfied_count = np.count_nonzero(latencies < target_slo)
    stats = {
        "target_slo": target_slo,
        "avg": np.nanmean(latencies),
        "min": np.nanmin(latencies),
        "max": np.nanmax(latencies),
        "p99": np.nanpercentile(latencies, 99),
        "p95": np.nanpercentile(latencies, 95),
        "p90": np.nanpercentile(latencies, 90),
        "slo_rate": satisfied_count/len(latencies)*100,
        "failed": np.count_nonzero(np.isnan(latencies)),
    }

    # SLO satisfaction rate per cluster
    # slo_counts = []
    slo_rates_dict = {}

    def calc_slo_rate(x):
        return x.sum()/len(x)
    for cluster_name, (tss, lats) in latency_dict.items():
        df = pd.DataFrame({"latency": lats, "ts": tss})
        if isinstance(target_slo, dict):
            cluster_slo = target_slo[cluster_name]
        else:
            cluster_slo = target_slo
        df["slo"] = (df.latency <= cluster_slo)
        df["bin"] = df.ts // unit_time
        slo_rate_per_bin = df.groupby(
            by="bin")["slo"].apply(calc_slo_rate).to_numpy()
        # stats[f"slo_rate_{cluster_name}"] = slo_rate_per_bin
        stats[f"avg_slo_rate_{cluster_name}"] = np.mean(slo_rate_per_bin)
        slo_rates_dict[cluster_name] = slo_rate_per_bin
    max_len = max([len(v) for v in slo_rates_dict.values()])
    new_slo_rate_dict = {}
    for k, v in slo_rates_dict.items():
        new_slo_rate_dict[k] = np.pad(
            v, (0, max_len - len(v)), constant_values=np.nan)
    slo_rate_df = pd.DataFrame(new_slo_rate_dict)
    stats["avg_total_slo_rate"] = slo_rate_df.mean(axis=1).mean()

    is_simulation = Path(target_dir / "sim_run.log").exists()
    if is_simulation:
        num_replicas_deployed_dict = get_num_deployed_replicas(
            target_dir, max_ts)
        # add the num replicas at the last moment since there is no line from the last to the end
        for l in num_replicas_deployed_dict.values():
            l.append((max_ts, l[-1][1]))

        num_replica_changes = calculate_total_replica_changes(
            num_replicas_deployed_dict)
        total_replica_changes = {ts: np.sum(v)
                                 for ts, v in num_replica_changes.items()}
        stats["avg_used_replicas"] = avg_used_replicas(total_replica_changes)
        stats["max_used_replicas"] = np.max(
            list(total_replica_changes.values()))

        # changes = utils.parse_operator_replica_change(
        #     target_dir, base_ts=0.0, max_ts=max_ts)
        markers = itertools.cycle(["o", "x", "^", "v", "<", ">"])

        out_path = target_dir.joinpath("replica_changes.png")
        if partial_end is not None:
            out_path = stats_path.with_name(f"{partial_end}_{out_path.name}")
        draw_replica_changes(out_path, num_replicas_deployed_dict,
                             total_replica_changes, markers)

    if unit_time is None:
        # simulation config has load specs
        if "load_spec" in config:
            unit_time = config["load_specs"][0]["params"].get("unit_time", 1)
        else:
            raise ValueError("--unit_time is required")

    count_dict = {}
    for cluster_name, (tss, _) in latency_dict.items():
        tss = np.array(tss)
        counts, _ = np.histogram(tss, bins=np.arange(
            np.ceil(np.max(tss) / unit_time)+1) * unit_time)
        count_dict[cluster_name] = counts

    if is_simulation:
        max_length = max([len(cluster_counts)
                          for cluster_counts in count_dict.values()])
        counts = np.sum(
            [np.pad(cluster_counts, (0, max_length - len(cluster_counts)))
             for cluster_counts in count_dict.values()], axis=0)
        out_path = target_dir.joinpath("replica_changes_input_count.png")
        if partial_end is not None:
            out_path = stats_path.with_name(f"{partial_end}_{out_path.name}")
        draw_replica_change_input_count(
            out_path, num_replicas_deployed_dict, counts, unit_time)

    if utility is not None:
        penalty_fn = SLOPenalty()

        def calculate_penalty(x):
            return penalty_fn(1 - np.sum(x)/len(x))

        stats["utility"] = utility
        # TODO: Fix hard coded lengths
        cluster_utilities = np.zeros(361)
        cluster_effective_utilities = np.zeros(361)
        quantile = 0.99
        for cluster_name, (tss, lats) in latency_dict.items():
            df = pd.DataFrame({"latency": lats, "ts": tss})
            df["bin"] = df.ts // unit_time
            new_df = df.fillna(np.nan_to_num(np.inf))
            latency_quantile = new_df[["latency", "bin"]].groupby(
                by="bin").quantile(quantile)
            if isinstance(target_slo, dict):
                cluster_slo = target_slo[cluster_name]
            else:
                cluster_slo = target_slo
            util_fn = np.vectorize(create_utility_func(utility, cluster_slo))
            utilities = latency_quantile.apply(util_fn)["latency"].values
            assert not np.any(np.isnan(utilities)), target_dir
            cluster_utilities[:len(utilities)] += utilities
            stats[f"avg_util_{cluster_name}"] = np.mean(utilities)

            # effective utility
            df["succeeded"] = ~df.latency.isna()
            penalties = df.groupby(by="bin").succeeded.apply(
                calculate_penalty).values
            # remove dropped requests
            latency_quantile = df[~df.latency.isna()][["latency", "bin"]].groupby(
                by="bin").quantile(quantile)
            utilities = latency_quantile.apply(util_fn)["latency"].values
            assert not np.any(np.isnan(utilities))
            min_len = min(len(penalties), len(utilities))
            effective_utilities = (
                1 - penalties[:min_len]) * utilities[:min_len]
            cluster_effective_utilities[:len(
                effective_utilities)] += effective_utilities  # [:360]
            stats[f"avg_effective_util_{cluster_name}"] = np.mean(
                effective_utilities)

        stats["avg_total_utility"] = np.mean(cluster_utilities)
        stats["avg_total_effective_utility"] = np.mean(
            cluster_effective_utilities)

        # fig, ax = plt.subplots()
        # avg_utils = calculate_aggregate(
        #     itertools.chain(*utilities), itertools.chain(*tss), unit_time)
        # ax.plot(avg_utils)
        # ax.set_xlabel(f"time (unit: {'s' if unit_time == 1 else 'm'})")
        # ax.set_ylabel("avg utility")
        # out_path = target_dir / "avg_utilities.png"
        # if partial_end is not None:
        #     out_path = stats_path.with_name(f"{partial_end}_{out_path.name}")
        # fig.savefig(out_path, bbox_inches="tight")

    pd.to_pickle(stats, stats_path)

    print(
        f"Done analyzing {target_dir}, elapsed_time: {time.time() - analysis_start_ts:.2f}", flush=True)
    return stats, target_dir


def draw_replica_change_input_count(out_path, num_replicas_deployed_dict, counts, unit_time):
    fig, ax = plt.subplots()  # (figsize=(10, 5))
    num_replica_changes = calculate_total_replica_changes(
        num_replicas_deployed_dict)
    ax.step(*zip(*[(ts, np.sum(v))
            for ts, v in num_replica_changes.items()]), label="total", where="post")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("num replicas")

    ax2 = ax.twinx()
    ax2.stairs(counts, np.arange(len(counts)+1) * unit_time,
               color="grey", label="input counts", alpha=0.8)
    ax2.set_ylabel("input counts")

    ax.set_title("# replica changes & input counts (total)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir")
    parser.add_argument("--include", action="append")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--utility", help="calculate utility stats")
    parser.add_argument("--force", action="store_true",
                        help="regenerate stats.pkl even if it exists.")
    parser.add_argument("--exclude", action="append")
    parser.add_argument("--prefix")
    parser.add_argument("--out-dir")
    parser.add_argument("--split-intense", action="store_true")
    parser.add_argument("--partial-end", type=int)
    parser.add_argument("--unit_time", type=int)

    args = parser.parse_args()

    run_script_path = Path(__file__).parent.joinpath("run.py")
    print("Run script:" + str(run_script_path), flush=True)
    assert run_script_path.exists()

    target_dir = Path(args.target_dir)
    target_config_dirs = []
    for target_config_path in glob((target_dir / "**/config.yaml").as_posix(), recursive=True):
        skip = False
        if args.include is not None:
            for include_str in args.include:
                if include_str not in str(target_config_path):
                    skip = True
                    break
        if not skip and args.exclude is not None:
            for exclude_str in args.exclude:
                if exclude_str in str(target_config_path):
                    skip = True
        if skip:
            continue
        target_config_dirs.append(Path(target_config_path).parent)

    assert len(target_config_dirs) > 0
    # random.shuffle(target_config_dirs)

    cuda_iterator = itertools.cycle([0, 1, 2, 3])

    def run_simulation(config_dir, out_dir, force):
        print(
            f"Run simluation: {str(config_dir)}, out_dir: {str(out_dir)}", flush=True)
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = ""  # str(next(cuda_iterator))
        # my_env["CUDA_LAUNCH_BLOCKING"] = "1"
        cmds = ["python", str(run_script_path), str(config_dir)]
        metric_path = Path(config_dir) / "metrics.pkl.gz"
        if out_dir is not None:
            cmds.append(f"--out_dir={out_dir}")
            metric_path = Path(out_dir) / "metrics.pkl.gz"
        if not force and metric_path.exists():
            print(
                f"Skip simluation: {str(config_dir)}. output exists at {str(out_dir)}.", flush=True)
        else:
            start_ts = time.time()
            subprocess.check_call(
                cmds,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                env=my_env,
            )
            print(
                f"Done simluation: {str(config_dir)}, elapsed_time: {time.time() - start_ts:.2f}", flush=True)

    if not args.stats:
        with tqdm(total=len(target_config_dirs)) as pbar:
            # not darts configs
            normal_config_dir = []
            intensive_config_dir = []
            for config_dir in target_config_dirs:
                config_dir = config_dir.as_posix()
                out_dir = None
                if args.out_dir is not None:
                    out_dir = args.out_dir
                    assert args.prefix is not None
                    if not out_dir.endswith("/"):
                        out_dir += "/"
                    begin = config_dir.find(args.prefix)
                    assert begin > 0
                    out_dir += config_dir[begin+len(args.prefix):]
                    out_dir = Path(out_dir).resolve().as_posix()
                if args.split_intense and ("dart" in config_dir or "gluonts" in config_dir):
                    intensive_config_dir.append((config_dir, out_dir))
                else:
                    normal_config_dir.append((config_dir, out_dir))
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [
                    executor.submit(run_simulation, config_dir,
                                    out_dir, args.force)
                    for config_dir, out_dir in normal_config_dir
                ]
                for future in as_completed(futures):
                    exception = future.exception()
                    if exception:
                        raise exception
                    future.result()
                    pbar.update(1)
            with ThreadPoolExecutor(max_workers=args.max_workers // 4) as executor:
                futures = [
                    executor.submit(run_simulation, config_dir,
                                    out_dir, args.force)
                    for config_dir, out_dir in intensive_config_dir
                ]
                for future in as_completed(futures):
                    exception = future.exception()
                    if exception:
                        raise exception
                    future.result()
                    pbar.update(1)
    else:
        stats_dict = {}
        with tqdm(total=len(target_config_dirs)) as pbar:
            with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [
                    executor.submit(
                        analyze_result,
                        target_dir=config_dir,
                        force=args.force,
                        utility=args.utility,
                        partial_end=args.partial_end,
                        unit_time=args.unit_time)
                    for config_dir in target_config_dirs
                ]
                for future in as_completed(futures):
                    exception = future.exception()
                    if exception:
                        raise exception
                    stats, config_dir = future.result()
                    stats_dict[config_dir] = stats
                    pbar.update(1)

        out_path = target_dir.joinpath("stats.pkl")
        if args.utility is not None:
            out_path = out_path.with_name(f"{args.utility}_{out_path.name}")
        if args.partial_end is not None:
            out_path = out_path.with_name(
                f"{args.partial_end}_{out_path.name}")
        with open(out_path, "wb") as f:
            pickle.dump(stats_dict, f)

        # transposed = defaultdict(list)

        # for path, stats in stats_dict.items():
        #     transposed["path"].append(path)
        #     for k, v in stats.items():
        #         transposed[k].append(v)

        # def to_str(l):
        #     return [str(v) for v in l]

        # def float_to_str(l):
        #     return [f"{v:.2f}" for v in l]

        # print(" ".join(to_str(transposed["path"])))
        # print(" ".join(float_to_str(transposed["avg"])))
        # print(" ".join(float_to_str(transposed["min"])))
        # print(" ".join(float_to_str(transposed["max"])))
        # print(" ".join(float_to_str(transposed["p99"])))
        # print(" ".join(float_to_str(transposed["p95"])))
        # print(" ".join(float_to_str(transposed["p90"])))
        # print("% ".join(float_to_str(transposed["slo_rate"])) + "%")
        # print(" ".join(float_to_str(transposed["slo_max_min_diff"])))
        # print("% ".join(float_to_str(transposed["slo_rate_cluster1"])) + "%")
        # print("% ".join(float_to_str(transposed["slo_rate_cluster2"])) + "%")
        # print("% ".join(float_to_str(transposed["slo_rate_cluster3"])) + "%")
        # print("% ".join(float_to_str(transposed["slo_rate_cluster4"])) + "%")
        # print("% ".join(float_to_str(transposed["slo_rate_cluster5"])) + "%")
        # print("% ".join(float_to_str(transposed["slo_rate_cluster6"])) + "%")
        # print(" ".join(float_to_str(transposed["avg_used_replicas"])))
        # print(" ".join(float_to_str(transposed["max_used_replicas"])))
