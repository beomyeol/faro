import json
from typing import List, Union
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_root_dir_path():
    return Path(__file__).resolve().parents[1]


def get_aggregated_input_counts(input_path: Path) -> List[int]:
    counts = []
    with open(input_path, "r") as f:
        for line in f.readlines():
            obj = json.loads(line)
            counts.append(sum(obj["urls"].values()))
    return counts


def get_mismatched_entry_count(start_times, counts):
    hist, bins = np.histogram(start_times, np.arange(0, max(start_times)+1, 1))
    mismatch_entry_cnt = 0
    for i, (result, input) in enumerate(zip(hist, counts)):
        if result != input:
            print(f"{i}: ({result}, {input})")
            mismatch_entry_cnt += 1
    return mismatch_entry_cnt


def load_results(dirpath: Path):
    out_path = dirpath.joinpath("out.log")
    if out_path.exists():
        latencies = []
        start_times = []
        urls = []
        with open(out_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    latencies.append(float(row["latency"]))
                except Exception as e:
                    print(row)
                    raise e
                start_times.append(float(row["start_time"]))
                urls.append(row["url"])
        return np.array(latencies), np.array(start_times), urls
    else:
        latency_path = dirpath.joinpath("latency.log")
        with open(latency_path, "r") as f:
            latencies = np.array([float(latency) for latency in f])
        start_time_path = dirpath.joinpath("start_time.log")
        with open(start_time_path, "r") as f:
            start_times = [float(start_ts) for start_ts in f]
        return latencies, start_times, None


def calculate_average_lats_per_second(latencies, start_times):
    lats_dict = defaultdict(list)
    for lat, ts in zip(latencies, start_times):
        int_ts = int(ts)
        lats_dict[int_ts].append(lat)

    out = np.zeros(max(lats_dict.keys())+1)
    for ts, lats in lats_dict.items():
        out[ts] = np.mean(lats)/1000
    return out


def calculate_aggregate(values, tss, unit_ts=1.0, aggregate=np.mean):
    value_lists = defaultdict(list)
    for value, ts in zip(values, tss):
        target_bin = int(ts / unit_ts)
        value_lists[target_bin].append(value)
    out = np.zeros(max(value_lists.keys())+1)
    for ts, values in value_lists.items():
        out[ts] = aggregate(values)
    return out


def calculate_effective_utility(utilities, tss, unit_ts, penalty_fn):
    df = pd.DataFrame({"tss": tss, "utility": utilities,
                      "succeeded": ~np.isnan(utilities)})
    groupby_obj = df.groupby(df['tss'].apply(lambda x: int(x // unit_ts)))
    mean_utility = groupby_obj.utility.mean()
    grouped_penalty = groupby_obj.succeeded.apply(
        lambda x: penalty_fn(1 - np.sum(x)/len(x)))
    return mean_utility * (1 - grouped_penalty)


def parse_operator_ts(line: str, base_ts: Union[datetime, float]) -> float:
    if line.startswith("["):
        ts = datetime.strptime(
            line[1: line.find("]")], "%Y-%m-%d %H:%M:%S,%f")
        return (ts - base_ts).total_seconds()
    else:
        # simulation result
        begin = line.find("[", line.find("[") + 1)
        if begin == -1:
            raise ValueError("No sim time")
        end = line.find("]", begin)
        return float(line[begin+1:end])


def parse_resources_str(resources_str: str):
    cpu_begin = resources_str.find("cpu=") + 4
    cpu_end = resources_str.find(",", cpu_begin)
    cpu = float(resources_str[cpu_begin:cpu_end])
    memory = float(resources_str[resources_str.rfind("memory=") + 7:-1])
    return {"cpu": cpu, "memory": memory}


def parse_operator_replica_change(result_dir: Path, base_ts: Union[datetime, float], max_ts) -> defaultdict:
    changes = defaultdict(list)
    sim_log_path = result_dir.joinpath("sim_run.log")
    is_simluation = False
    if sim_log_path.exists():
        is_simluation = True
        lines = sim_log_path.read_text().splitlines()
    else:
        log_path = result_dir.joinpath("operator.log")
        if not log_path.exists():
            print("Operator log does not exist")
            return changes
        lines = log_path.read_text().splitlines()
        assert abs(parse_operator_ts(lines[0], base_ts)) < (
            20 * 60)  # timedelta(minutes=20)
    pattern = "Changing num_replicas for ["
    done_pattern = "Changing replica for ["
    if "kopf" in lines[0]:
        # k8s only autoscaler
        pattern = "Change quota for "
        for line in lines:
            if pattern in line:
                begin = line.find(pattern)+len(pattern)
                cluster_name = line[begin:line.find(" from", begin)]
                cluster_name = cluster_name[1+len("serve-"):-1]  # remove ''
                ts = parse_operator_ts(line, base_ts)
                if ts > max_ts:
                    continue
                begin = line.find("Resources")
                prev_quota = parse_resources_str(line[begin:line.find(")")-1])
                new_quota = parse_resources_str(line[line.rfind("Resources"):])
                changes[cluster_name].append((ts, prev_quota, new_quota))
    else:
        for line in lines:
            if pattern in line:
                ts = parse_operator_ts(line, base_ts)
                if ts > max_ts:
                    continue
                begin = line.find(pattern) + len(pattern)
                if is_simluation:
                    end = line.find("]", begin)
                else:
                    begin += len("serve-")
                    end = line.find("-ray-head", begin)
                cluster_name = line[begin:end]
                prev = int(line[line.rfind("from")+5:line.rfind("to")].strip())
                new = int(line[line.rfind("to")+2:].strip())
                changes[cluster_name].append((ts, prev, new))
            elif done_pattern in line:
                ts = parse_operator_ts(line, base_ts)
                if ts > max_ts:
                    continue
                begin = line.find(done_pattern) + len(done_pattern)
                if is_simluation:
                    end = line.find("]", begin)
                else:
                    begin += len("serve-")
                    end = line.find("-ray-head", begin)
                cluster_name = line[begin:end]
                changes[cluster_name].append((ts, None, None))
    return changes


def get_first_post_ts(result_dir):
    ts_dict = {}

    serve_dir = result_dir.joinpath("serve")
    if serve_dir.exists():
        # get min ts
        for cluster_dir in serve_dir.iterdir():
            proxy = list(cluster_dir.glob("http_proxy_*.log"))[0]
            # read only first line
            for line in proxy.read_text().splitlines():
                if "POST" in line:
                    time_str = " ".join(line.split(" ")[1:3])
                    ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                    ts_dict[cluster_dir.name] = ts
                    break
    else:
        for head_log in result_dir.glob("*head.log"):
            lines = head_log.read_text().splitlines()
            for line in lines:
                if "POST" in line:
                    time_str = " ".join(line.split(" ")[1:3])
                    ts_dict[head_log.name.split(".")[0]] = datetime.strptime(
                        time_str, "%Y-%m-%d %H:%M:%S,%f")
                    break
    return ts_dict


def draw_input_count(start_times, ax=None, label=""):
    if ax is None:
        ax = plt.subplot()
    count, bin_edges = np.histogram(
        start_times, bins=np.arange(0, max(start_times)+1, 1))
    ax.stairs(count, bin_edges, label=label)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("num requests")
    return ax


def get_cluster_latency(result_dir):
    serve_dir = result_dir.joinpath("serve")
    if serve_dir.exists():
        min_ts = None
        latency_dict = {}
        for cluster_dir in serve_dir.iterdir():
            proxy = list(cluster_dir.glob("http_proxy_*.log"))[0]
            tss = []
            lats = []
            for line in proxy.read_text().splitlines():
                if "POST" not in line:
                    continue
                split = line.split(" ")
                time_str = " ".join(split[1:3])
                ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                if min_ts is None:
                    min_ts = ts
                else:
                    min_ts = min(ts, min_ts)
                latency = split[-1]
                if latency.rfind("ms") == -1:
                    continue
                latency_ms = float(latency[:latency.rfind("ms")])
                tss.append(ts)
                lats.append(latency_ms)
            latency_dict[cluster_dir.name] = (tss, lats)

        return {
            cluster_name: ([(ts - min_ts).total_seconds() for ts in tss], lats)
            for cluster_name, (tss, lats) in latency_dict.items()
        }
    else:
        return None


def get_first_post_ts(result_dir):
    ts_dict = {}

    serve_dir = result_dir.joinpath("serve")
    if serve_dir.exists():
        # get min ts
        for cluster_dir in serve_dir.iterdir():
            proxy = list(cluster_dir.glob("http_proxy_*.log"))[0]
            # read only first line
            for line in proxy.read_text().splitlines():
                if "POST" in line:
                    time_str = " ".join(line.split(" ")[1:3])
                    ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                    ts_dict[cluster_dir.name] = ts
                    break
    else:
        for head_log in result_dir.glob("*head.log"):
            lines = head_log.read_text().splitlines()
            for line in lines:
                if "POST" in line:
                    time_str = " ".join(line.split(" ")[1:3])
                    ts_dict[head_log.name.split(".")[0]] = datetime.strptime(
                        time_str, "%Y-%m-%d %H:%M:%S,%f")
                    break
    return ts_dict


def get_replica_deployments_dict(result_dir):
    replica_deployments_dict = {}

    serve_dir = result_dir.joinpath("serve")
    if serve_dir.exists():
        print("Reading from serve...")
        min_ts = None
        # get min ts
        for cluster_dir in serve_dir.iterdir():
            proxy = list(cluster_dir.glob("http_proxy_*.log"))[0]
            for line in proxy.read_text().splitlines():
                if "POST" in line:
                    split = line.split(" ")
                    time_str = " ".join(split[1:3])
                    ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                    if min_ts is None:
                        min_ts = ts
                    else:
                        min_ts = min(ts, min_ts)

            replica_deployments = []
            for controller_fpath in cluster_dir.glob("controller_*.log"):
                lines = controller_fpath.read_text().splitlines()
                for line in lines:
                    if "replicas" in line and "controller" in line:
                        time_str = " ".join(line.split(" ")[1:3])
                        ts = datetime.strptime(
                            time_str, "%Y-%m-%d %H:%M:%S,%f")
                        replica_deployments.append((ts, line.split(" - ")[-1]))
            replica_deployments_dict[cluster_dir.name] = replica_deployments
        print("min_ts:", min_ts)

        replica_deployments_dict = {
            cluster_name: [((ts - min_ts).total_seconds(), log)
                           for ts, log in replica_deployments
                           if ts > min_ts and "have taken more than" not in log]
            for cluster_name, replica_deployments in replica_deployments_dict.items()
        }
    else:
        print("Reading from head.log...")
        for head_log in result_dir.glob("*head.log"):
            print(head_log.name)
            lines = head_log.read_text().splitlines()
            # find add replica times
            first_post_ts = None
            replica_deployments = []
            for line in lines:
                if "replicas" in line and "controller" in line:
                    time_str = " ".join(line.split(" ")[1:3])
                    ts = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S,%f")
                    replica_deployments.append((ts, line.split(" - ")[-1]))
                if "POST" in line and first_post_ts is None:
                    time_str = " ".join(line.split(" ")[1:3])
                    first_post_ts = datetime.strptime(
                        time_str, "%Y-%m-%d %H:%M:%S,%f")
            replica_deployments = [
                ((ts - first_post_ts).total_seconds(), log)
                for ts, log in replica_deployments
                if ts > first_post_ts]
            replica_deployments = [
                (ts, log) for ts, log in replica_deployments
                if "have taken more than" not in log
            ]
            if len(replica_deployments) > 0:
                replica_deployments_dict[head_log.name.split(
                    ".")[0]] = replica_deployments

    return replica_deployments_dict


def draw_replica_deployment_lines(replica_deployments_dict, ax):
    lines = []
    for name, replica_deployments in replica_deployments_dict.items():
        for ts, log in replica_deployments:
            lines.append(ax.axvline(x=ts, label=f"{int(ts)}:{log}"))
    return lines


def remove_out_of_scope(replica_deployments_dict, max_time):
    for replica_deployments in replica_deployments_dict.values():
        targets = []
        for i, (ts, log) in enumerate(replica_deployments):
            if ts > max_time:
                targets.append(i)
        for target in targets:
            print("Removing '{}'".format(replica_deployments[target]))
            del replica_deployments[target]
    return replica_deployments_dict


def print_stats(latencies):
    print(f"count: {len(latencies)}")
    print(f"avg: {np.mean(latencies):.0f}")
    print(f"min: {np.min(latencies):.0f}")
    print(f"max: {np.max(latencies):.0f}")
    print(f"p99: {np.percentile(latencies, 99):.0f}")
    print(f"p95: {np.percentile(latencies, 95):.0f}")
    print(f"p90: {np.percentile(latencies, 90):.0f}")
    satiesfied_count = np.count_nonzero(latencies < 800)
    print(
        f"count < 0.8s: {satiesfied_count} ({satiesfied_count/len(latencies)*100:.2f}%)")
    print(f"count > 2s: {np.count_nonzero(latencies > 2000)}")


def generate_num_replicas(counts, window_size, interval, concurrent_num_reqs):
    split = []
    for idx in np.arange(0, len(counts), interval):
        s = counts[idx:idx+window_size]
        if len(s) < window_size:
            continue
        split.append(s)
    split = np.stack(split)
    print(split.shape)
    num_replicas = np.ceil(np.average(split, axis=1) / concurrent_num_reqs)

    num_changes = 0
    current_num_replicas = 1
    for n in num_replicas:
        if n != current_num_replicas:
            current_num_replicas = n
            num_changes += 1
    print("max_num_replicas:", max(num_replicas))
    print("num_changes:", num_changes)
    print("avg num replicas:", np.mean(num_replicas))
    return num_replicas


def get_num_replicas_demand(result_dir: Path, base_ts, max_ts):
    log_path = result_dir.joinpath("operator.log")
    num_replicas = []
    if not log_path.exists():
        print("Operator log does not exist")
        return num_replicas
    lines = log_path.read_text().splitlines()
    # timedelta(minutes=20)
    assert abs(parse_operator_ts(lines[0], base_ts)) < 20 * 60
    pattern = "num_replicas_demand: "
    for line in lines:
        if pattern in line:
            ts = parse_operator_ts(line, base_ts)
            if ts > max_ts:
                continue
            num_replicas.append(
                (ts, int(line[line.find(pattern)+len(pattern):])))
    return num_replicas


def get_num_replica_history(changes, start_num_replica=1):
    num_replicas_decision = []
    num_replicas_deployed = []

    current_num_replica = start_num_replica
    num_replicas_decision.append((0, current_num_replica))
    num_replicas_deployed.append((0, current_num_replica))

    for ts, prev_num, new_num in changes:
        if prev_num is None:
            num_replicas_deployed.append((ts, current_num_replica))
        else:
            current_num_replica = new_num
            num_replicas_decision.append((ts, new_num))
    return num_replicas_decision, num_replicas_deployed


def print_num_replica_stats(num_replica_list):
    num_replicas = np.array(
        [num_replica for _, num_replica in num_replica_list])
    print("num changes:", len(num_replicas))
    print("max num replicas:", np.max(num_replicas))
    print("min num replicas:", np.min(num_replicas))
    print("avg num replicas:", np.mean(num_replicas))


def draw_change_lines(changes: list):
    lines = []
    for ts, prev, new in sorted(changes, key=lambda x: x[0]):
        if isinstance(prev, dict):
            label = f"{int(ts)}:cpu=({prev['cpu']}->{new['cpu']})"
            color = "r" if prev['cpu'] < new['cpu'] else "b"
        elif prev is None:
            # changing replicas done log
            # append ts at the previous line label
            prev_line = lines[-1]
            prev_line.set_label(prev_line.get_label() + f"[{int(ts)}]")
            continue
            # quota_changes.values()
            # label = f"{int(ts)}:done"
        else:
            label = f"{int(ts)}:#replica={prev}->{new}"
            color = "r" if prev < new else "b"
        lines.append(plt.axvline(x=ts, label=label, color=color))
    return lines


def parse_utilization_info(result_dir: Path, base_ts: datetime):
    lines = result_dir.joinpath("operator.log").read_text().splitlines()
    # timedelta(minutes=20)
    assert abs(parse_operator_ts(lines[0], base_ts)) < 20 * 60
    util_dict = defaultdict(list)
    for line in lines:
        if "Utilization for pod" not in line:
            continue
        ts = parse_operator_ts(line, base_ts)
        pod_name = line[line.find("pod ") + 4: line.rfind(":")]
        resources = parse_resources_str(line[line.rfind(":"):])
        util_dict[pod_name].append((ts, resources))
    return util_dict


def parse_avg_util_info(result_dir: Path, base_ts: datetime):
    lines = result_dir.joinpath("operator.log").read_text().splitlines()
    # timedelta(minutes=20)
    assert abs(parse_operator_ts(lines[0], base_ts)) < 20 * 60
    util_dict = defaultdict(list)
    pattern = "Avg utilization for cluster serve-"
    for line in lines:
        if not pattern in line:
            continue
        ts = parse_operator_ts(line, base_ts)
        cluster_name = line[line.find(pattern) + len(pattern): line.rfind(":")]
        resources = parse_resources_str(line[line.rfind(":"):])
        util_dict[cluster_name].append((ts, resources))
    return util_dict


def parse_admission_info(result_dir: Path, base_ts: datetime):
    operator_log_path = result_dir.joinpath("operator.log")
    if not operator_log_path.exists():
        return None
    lines = operator_log_path.read_text().splitlines()
    if len(lines) == 0:
        return None
    if "kopf" not in lines[0]:
        return None
    # print(lines[0])
    # print(base_ts)
    # print(abs(parse_operator_ts(lines[0], base_ts)))
    # timedelta(minutes=20)
    assert abs(parse_operator_ts(lines[0], base_ts)) < 20 * 60
    admission_dict = defaultdict(list)
    start_pattern = "[k8s-ray/"
    for line in lines:
        if "Webhook" in line and line.endswith("succeeded."):
            begin = line.find(start_pattern)+len(start_pattern)
            pod_name = line[begin:line.find("]", begin)]
            begin = pod_name.find("cluster")
            cluster_name = pod_name[begin:pod_name.find("-", begin)]
            ts = parse_operator_ts(line, base_ts)
            if ts < 0:
                continue
            admission_dict[cluster_name].append((ts, pod_name))
    return admission_dict


def draw_cpu_utilization(util_dict, ax: plt.Axes, max_ts):
    lines = []
    for pod_name, util_infos in sorted(util_dict.items()):
        tss = []
        cpu_utils = []
        for ts, resources in util_infos:
            if ts > max_ts:
                continue
            tss.append(ts)
            cpu_utils.append(resources["cpu"])
        label = pod_name[pod_name.rfind("serve"):]
        lines.extend(ax.plot(tss, cpu_utils, label=label))
    return lines
