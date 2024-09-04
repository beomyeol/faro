import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


def parse_latencies(target_dir, with_worker: bool):
    clusters = []
    methods = []
    status_codes = []
    lats = []
    end_tss = []

    for proxy_log_path in Path(target_dir).glob("**/http_proxy_*.log"):
        cluster_name = proxy_log_path.parents[1].name
        for line in proxy_log_path.read_text().splitlines():
            if "POST" in line:
                split = line.split(" ")
                end_ts = datetime.strptime(" ".join(split[1:3]), "%Y-%m-%d %H:%M:%S,%f")
                line = line[line.rfind("-")+1:]
                split = line.strip().split(" ")
                latency = float(split[-1][:-2])
                status_code = split[-2]
                clusters.append(cluster_name)
                methods.append(split[0])
                status_codes.append(status_code)
                lats.append(latency)
                end_tss.append(end_ts.timestamp())

    if with_worker:
        for replica_log_path in Path(target_dir).glob("**/deployment_classifier*.log"):
            cluster_name = replica_log_path.parents[1].name
            for line in replica_log_path.read_text().splitlines():
                if "HANDLE" in line:
                    split = line.split(" ")
                    end_ts = datetime.strptime(" ".join(split[1:3]), "%Y-%m-%d %H:%M:%S,%f")
                    line = line[line.rfind("-")+1:]
                    split = line.strip().split(" ")
                    latency = float(split[-1][:-2])
                    status_code = split[-2]
                    clusters.append(cluster_name)
                    methods.append(split[0])
                    status_codes.append(status_code)
                    lats.append(latency)
                    end_tss.append(end_ts.timestamp())

    df = pd.DataFrame({
        "cluster": clusters,
        "method": methods,
        "status": status_codes,
        "latency_ms": lats,
        "end_ts": end_tss,
    })

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--out", default="metrics.pkl.gz")
    parser.add_argument("--with-worker", action="store_true")

    args = parser.parse_args()

    df = parse_latencies(args.log_dir, args.with_worker)

    print("[Post]")
    post_df = df[df.method == "POST"]
    success_post_df = post_df[post_df.status == "200"]
    failed_post_df = post_df[post_df.status == "503"]

    print("\tcount", len(success_post_df))
    print("\tp90", success_post_df.latency_ms.quantile(0.9))
    print("\taverage", success_post_df.latency_ms.mean())
    print("\tmedian", success_post_df.latency_ms.median())
    print("\tfailed_count", len(failed_post_df))
    print("\ttotal_count", len(failed_post_df) + len(success_post_df))

    if args.with_worker:
        handle_df = df[df.method == "HANDLE"]
        success_handle_df = handle_df[handle_df.status == "OK"]

        overheads = (
            success_post_df.latency_ms.reset_index(drop=True) -
            success_handle_df.latency_ms.reset_index(drop=True))

        print("[Handle]")
        print("\tcount", len(success_handle_df))
        print("\tp90", success_handle_df.latency_ms.quantile(0.9))
        print("\taverage", success_handle_df.latency_ms.mean())
        print("\tmedian", success_handle_df.latency_ms.median())

        print("[Overhead]")
        print("\tp90", overheads.quantile(0.9))
        print("\taverage", overheads.mean())
        print("\tmedian", overheads.median())

    # convert to metric df format.
    df["cluster_name"] = df["cluster"]
    df["arrival_ts"] = df["end_ts"] - df["latency_ms"] / 1e3
    min_arrival_ts = df["arrival_ts"].min()
    df["arrival_ts"] -= min_arrival_ts
    df["end_ts"] -= min_arrival_ts
    df["succeeded"] = (df["status"] == "200")

    output_path = Path(args.log_dir) / args.out
    print(f"Writing output file to {output_path}...")
    df.to_pickle(output_path)
