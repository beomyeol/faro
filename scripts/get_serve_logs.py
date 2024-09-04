import argparse
import logging
from pathlib import Path
import subprocess
import sys


_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    format = '%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=format)

    parser = argparse.ArgumentParser()
    parser.add_argument("output_path")
    parser.add_argument("--with-worker", action="store_true")
    parser.add_argument("--with-autoscaler", action="store_true")

    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir.joinpath("src").as_posix()
    sys.path.append(src_dir)
    import k8s_utils

    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    if args.with_autoscaler:
        pods = k8s_utils.list_autoscale_controller_pod()
        assert len(pods) > 0
        pod_name = pods[0].metadata.name
        k8s_utils.copy_from_pod(
            pod_name, "/home/ray/run.log", output_path / "run.log")
        k8s_utils.copy_from_pod(
            pod_name, "/home/ray/config.yaml", output_path / "config.yaml")

    head_pod_dict = k8s_utils.list_ray_head_pods()
    for cluster_name, head_pod in head_pod_dict.items():
        cluster_path = output_path.joinpath(cluster_name)
        cluster_path.mkdir()
        local_path = cluster_path.joinpath(head_pod[head_pod.find("head-"):])
        local_path.mkdir()
        k8s_utils.copy_from_pod(
            head_pod, "/tmp/ray/session_latest/logs/serve", local_path)

    if args.with_worker:
        worker_pod_dict = k8s_utils.list_ray_worker_pods()
        for cluster_name, worker_pods in worker_pod_dict.items():
            cluster_path = output_path.joinpath(cluster_name)
            for worker_pod in worker_pods:
                local_path = cluster_path.joinpath(
                    worker_pod[worker_pod.find("worker-"):])
                local_path.mkdir()
                k8s_utils.copy_from_pod(
                    worker_pod, "/tmp/ray/session_latest/logs/serve", local_path)
