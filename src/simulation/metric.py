from __future__ import annotations
import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING, DefaultDict, Iterable, List
from pathlib import Path

import numpy as np

if TYPE_CHECKING:
    from simulation.primitives import Task

_TASKS: DefaultDict[str, List[TimeStampedTask]] = defaultdict(list)
_SCALE_OUT_ATTEMPTS = defaultdict(int)
_SCALE_OUT_COUNTS = defaultdict(int)
_SCALE_IN_COUNTS = defaultdict(int)
_AUTOSCALE_LATENCIES = []


def dump(path_str: str):
    import pickle
    out = {
        "tasks": _TASKS,
        "scale_out_attempts": _SCALE_OUT_ATTEMPTS,
        "scale_out_counts": _SCALE_OUT_COUNTS,
        "scale_in_counts": _SCALE_IN_COUNTS,
        "autoscale_latencies": _AUTOSCALE_LATENCIES,
    }
    path = Path(path_str)
    with open(path, "wb") as f:
        pickle.dump(out, f)


def to_df():
    import pandas as pd
    items = []
    for cluster_name, timestamped_tasks in _TASKS.items():
        for timestamped_task in timestamped_tasks:
            task = timestamped_task.task
            items.append(
                (cluster_name, task.arrival_ts, task.start_ts,
                 task.end_ts, task.succeeded)
            )
    cluster_names, arrival_tss, start_tss, end_tss, succeededs = zip(*items)
    return pd.DataFrame({
        "cluster_name": cluster_names,
        "arrival_ts": arrival_tss,
        "start_ts": start_tss,
        "end_ts": end_tss,
        "succeeded": succeededs,
    })

def dump_df(path_str: str):
    to_df().to_pickle(path_str)


def write_to_dir(path_str: str):
    dirpath = Path(path_str)

    dirpath.mkdir(exist_ok=True)

    for cluster_name, timestamped_tasks in _TASKS.items():
        times = []
        for timestamped_task in timestamped_tasks:
            task = timestamped_task.task
            times.append("%s %s" %
                         (task.arrival_ts, task.elapsed_time() * 1e3))
        out_path = dirpath.joinpath(f"time_{cluster_name}.log")
        out_path.write_text("\n".join(times))

    if len(_AUTOSCALE_LATENCIES) > 0:
        out_path = dirpath.joinpath("autoscale_latency.log")
        out_path.write_text("\n".join([str(v) for v in _AUTOSCALE_LATENCIES]))


@dataclass(order=True)
class TimeStampedTask:
    timestamp: float
    task: Task = field(compare=False)


def put(task: Task) -> None:
    # TODO: should the timestamp be start_ts or end_ts?
    bisect.insort(_TASKS[task.cluster_name],
                  TimeStampedTask(task.end_ts, task))


def get_tasks(cluster_name: str, start_ts: float = 0) -> List[TimeStampedTask]:
    ts_tasks = _TASKS[cluster_name]
    idx = bisect.bisect(ts_tasks, TimeStampedTask(start_ts, None))
    return ts_tasks[idx:]


def num_tasks(cluster_name=None) -> int:
    if cluster_name is not None:
        return len(_TASKS[cluster_name])
    return np.sum([len(tasks) for tasks in _TASKS.values()])


def _get_tasks(cluster_name=None) -> Iterable[Task]:
    if cluster_name is not None:
        return _TASKS[cluster_name]
    return chain(*_TASKS.values())


def get_wait_times(cluster_name=None, succeeded: bool = True) -> List[float]:
    return [
        ts_task.task.wait_time() for ts_task in _get_tasks(cluster_name)
        if ts_task.task.succeeded == succeeded
    ]


def get_total_times(cluster_name=None, succeeded: bool = True) -> List[float]:
    return [
        ts_task.task.elapsed_time() for ts_task in _get_tasks(cluster_name)
        if ts_task.task.succeeded == succeeded
    ]


def try_scale_out(cluster_name, count=1):
    _SCALE_OUT_ATTEMPTS[cluster_name] += count


def scale_out_done(cluster_name, count=1):
    _SCALE_OUT_COUNTS[cluster_name] += count


def scale_in_done(cluster_name, count=1):
    _SCALE_IN_COUNTS[cluster_name] += count


def autoscale_latency(elapsed_time: float):
    _AUTOSCALE_LATENCIES.append(elapsed_time)


def print_stats(target_latency: float = None):
    wait_times = np.array(get_wait_times())
    total_times = np.array(get_total_times())
    assert len(wait_times) == len(total_times)

    def _print_stats(times):
        print(f"\tmin: {np.min(times):.2f}")
        print(f"\tavg: {np.mean(times):.2f}")
        print(f"\tmedian: {np.median(times):.2f}")
        print(f"\tmax: {np.max(times):.2f}")

    total_num_tasks = num_tasks()
    print(f"# clusters: {len(_TASKS)}")
    print(f"# tasks: {total_num_tasks}")
    print(f"# succceed: {len(total_times)}")
    print(f"# failed: {total_num_tasks - len(total_times)}")
    print("[wait time]")
    _print_stats(wait_times * 1e3)
    print("[total time]")
    _print_stats(total_times * 1e3)

    if target_latency is not None:
        print("[SLO]")
        count = np.count_nonzero(total_times < target_latency)
        print("\t# tasks < target_latency (%.2f): %d (%.2f%%)" %
              (target_latency, count, (count/total_num_tasks*100)))

    print("[scale in/out]")
    print(f"\tout attempts: {sum(_SCALE_OUT_ATTEMPTS.values())}")
    print(f"\tout counts: {sum(_SCALE_OUT_COUNTS.values())}")
    print(f"\tin counts: {sum(_SCALE_IN_COUNTS.values())}")

    print("[autoscale]")
    print(f"\tcount: {len(_AUTOSCALE_LATENCIES)}")
    if len(_AUTOSCALE_LATENCIES) > 0:
        print(f"\tmin: {np.min(_AUTOSCALE_LATENCIES)}")
        print(f"\tmax: {np.max(_AUTOSCALE_LATENCIES)}")
        print(f"\tavg: {np.average(_AUTOSCALE_LATENCIES)}")
        print(f"\tmedian: {np.median(_AUTOSCALE_LATENCIES)}")
