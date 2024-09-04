import argparse
import itertools
from joblib import Parallel, delayed
import pandas as pd
import json
from pathlib import Path
import numpy as np
from collections import defaultdict
import sys

sys.path.append("../src")
from solver import create_solver, run_solver, _calculate_utility

def run_mark(factor):
    values = []
    for target in range(interval, len(df), 5):
        input_rates = df.iloc[target:target+window_size].values.T
        input_rates = np.concatenate(
            [input_rates for _ in range(factor)], axis=0)
        num_replicas = np.ones(input_rates.shape[0])
        # print(num_replicas)
        x = np.ceil(input_rates.max(axis=1) / 4)
        resource_limit = 32 * factor
        # print(x, x.sum())
        undeployable = int(x.sum() - resource_limit)
        for i in range(len(x) - 1, -1, -1):
            if undeployable <= 0:
                break
            if x[i] > 1:
                removable = min(x[i] - 1, undeployable)
                x[i] -= removable
                undeployable -= removable
        # print(x, x.sum())
        v = _calculate_utility(
            x.astype(np.float64), num_replicas.astype(np.float64), 1,
            input_rates, np.ones_like(x) * processing_time,
            base_solver_kwargs["mdc_percentile"],
            0.95, base_solver_kwargs["util_type"],
            np.ones_like(x) * target_latency)
        values.append(v.sum())
    return values


def run_single(factor, num_children=None, contiguous=False, random_split=False):
    values = []
    elapsed_times = []
    num_replicas = np.ones(len(df.columns) * factor, dtype=int)
    # print("num_replicas:", len(num_replicas))
    solver_kwargs = dict(base_solver_kwargs)
    solver_kwargs["resource_limit"] *= factor
    solver_kwargs["num_children"] = num_children
    if num_children is not None and num_children > 1:
        solver_kwargs["random_split"] = bool(random_split)

    for target in range(interval, len(df), 5):
        input_rates = df.iloc[target:target+window_size].values.T
        input_rates = np.concatenate(
            [input_rates for _ in range(factor)], axis=0)

        num_replicas, value, elapsed_time = run_solver(
            solver=create_solver(**solver_kwargs),
            input_rates=input_rates,
            num_replicas=num_replicas,
            processing_time=processing_time,
            target_latency=target_latency,
        )
        if not contiguous:
            num_replicas = np.ones(len(df.columns) * factor, dtype=int)

        values.append(value)
        elapsed_times.append(elapsed_time)

    return values, elapsed_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=7)
    parser.add_argument("--processing-time", type=int, default=180)
    parser.add_argument("--target-latency", type=int, default=720)
    parser.add_argument("--resource-limit", type=int, default=32)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--max-g", type=int)
    parser.add_argument("--max-factor", type=int)
    parser.add_argument("--contiguous", action="store_true",
                        help="use prev number of replicas as the next input")
    parser.add_argument("--random-split", type=int,
                        help="randomly split into groups with the given trial")

    args = parser.parse_args()

    input_rate_json = Path(args.input)
    data_dict = defaultdict(list)

    for line in input_rate_json.read_text().splitlines():
        obj = json.loads(line)
        for cluster_name, kv in obj["counts"].items():
            for dep_name, count in kv.items():
                data_dict[cluster_name].append(count)

    df = pd.DataFrame(data_dict)
    df = df / 60

    interval = args.interval
    window_size = args.window_size
    processing_time = args.processing_time
    target_latency = args.target_latency

    base_solver_kwargs = {
        'with_drop': False,
        "adjust": True,
        "min_max": False,
        "method": 'cobyla',
        "mdc_percentile": 99,
        "util_type": 'latency',
        "upscale_overhead": 1,
        "resource_limit": args.resource_limit,
    }

    values_dict = defaultdict(list)
    elapsed_times_dict = defaultdict(list)

    num_childrens = [1, 2, 3, 5, 10, 20]
    factors = list(range(1, 21))[::-1]

    if args.max_g is not None:
        num_childrens = [v for v in num_childrens if v <= args.max_g]
    if args.max_factor is not None:
        factors = [v for v in factors if v <= args.max_factor]

    configs = list(itertools.product(factors, num_childrens))
    configs = [config for config in configs if config not in values_dict]
    if args.random_split:
        print(f"Duplicate configs {args.random_split} with random split")
        new_configs = []
        for factor, num_children in configs:
            if num_children > 1:
                for _ in range(args.random_split):
                    new_configs.append((factor, num_children))
            else:
                new_configs.append((factor, num_children))
        configs = new_configs

    with Parallel(n_jobs=-1, backend='loky', verbose=10, return_as="generator") as parallel:
        outputs = parallel(
            delayed(run_single)(factor, num_children, args.contiguous, args.random_split)
            for factor, num_children in configs)
        for config, (values, elapsed_times) in zip(configs, outputs):
            values_dict[config].append(values)
            elapsed_times_dict[config].append(elapsed_times)

    # import logging
    # format = '%(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    # logging.basicConfig(
    #     level=logging.getLevelName(logging.INFO), format=format)
    # values, elapsed_times = run_single(10, 1)

    if args.out_path is not None:
        output_path = args.out_path
    else:
        output_path = f"sum_{max(num_childrens)}group_{max(factors)}factor"
        if not args.contiguous:
            output_path += "_from1"
        if args.random_split:
            output_path += f"_rand{args.random_split}"
        if args.resource_limit != 32:
            output_path += f"_resource{args.resource_limit}"
        output_path += ".pkl"

    print("Wrinting to ", output_path)

    pd.to_pickle(
        {
            "values_dict": values_dict,
            "elapsed_times_dict": elapsed_times_dict,
            "base_solver_kwargs": base_solver_kwargs,
        },
        output_path
    )
