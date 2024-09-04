from typing import List, Dict
import subprocess
import logging

_LOGGER = logging.getLogger(__name__)

def get_head_pods() -> List[str]:
    pods = subprocess.check_output(
        ["kubectl", "get", "pods", "-n", "k8s-ray"]).decode("utf-8")
    head_pods = [
        line.split()[0] for line in pods.splitlines() if "ray-head" in line]
    return head_pods


def get_cluster_ips() -> Dict[str, str]:
    pods = subprocess.check_output(
        ["kubectl", "get", "services", "-n", "k8s-ray"]).decode("utf-8")
    cluster_ips = {}
    for line in pods.splitlines():
        if "ray-head" in line:
            split = line.split()
            cluster_ips[split[0]] = split[2]
    return cluster_ips


def copy_to_pod(local_path: str, pod_name: str, target_path: str):
    pod_path = f"{pod_name}:{target_path}"
    _LOGGER.info("Copying %s to %s...", local_path, pod_path)
    subprocess.check_call([
        "kubectl", "cp", "-n", "k8s-ray", str(local_path), pod_path
    ])
