import argparse
from pathlib import Path
import subprocess
import time
import sys

def copy_to_pod(local_path: str, pod_name: str, target_path: str):
    pod_path = f"{pod_name}:{target_path}"
    print("Copying {} to {}...".format(local_path, pod_path), flush=True)
    subprocess.check_call([
        "kubectl", "cp", "-n", "k8s-ray", str(local_path), pod_path
    ])

def subproc_run(args):
    out = subprocess.run(args, shell=True, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8"))
    result = out.stdout.decode("utf-8").strip()
    return result

def get_worker_pods(cluster_name):
    pods = subprocess.check_output(
        ["kubectl", "get", "pods", "-n", "k8s-ray"]).decode("utf-8")
    worker_pods = [
        line.split()[0] for line in pods.splitlines()
        if cluster_name in line and "worker" in line and "Running" in line
    ]
    return worker_pods

def wait_for_n_workers(cluster_name, num):
    while True:
        print("Waiting for {} workers of {}".format(num, cluster_name), flush=True)
        worker_pods = get_worker_pods(cluster_name)
        if len(worker_pods) >= num:
            break
        time.sleep(3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--num_replicas", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    cluster_name = args.target
    script_dir = Path(__file__).parent.absolute()
    deploy_script_path = script_dir.joinpath("train.py")

    cmd = f"python -c \"import ray;ray.init('auto');ray.autoscaler.sdk.request_resources(num_cpus={args.num_replicas})\""

    if args.num_replicas > 1:
        wait_for_n_workers(cluster_name, 1)
    subproc_run(f"kubectl -n k8s-ray exec --stdin --tty svc/{cluster_name}-head-svc -c ray-head -- {cmd}")
    wait_for_n_workers(cluster_name, args.num_replicas - 1)  # -1 for head

    cmd = f"kubectl get pods -n k8s-ray -o wide | grep {cluster_name}-head"
    cmd += " | awk '{print $1, $6}'"
    head_pod = subproc_run(cmd)
    head_pod, head_ip = head_pod.split(" ")
    assert len(head_pod) > 0

    copy_to_pod(deploy_script_path, head_pod, deploy_script_path.name)

    cmd = f"python train.py --epochs={args.epochs}"
    proc = subprocess.Popen(
        f"kubectl exec -n k8s-ray --tty pod/{head_pod} -- {cmd}", 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, shell=True)
    for line in iter(proc.stdout.readline, b''):
        print(line.rstrip().decode('utf-8'), flush=True)
    proc.wait()
