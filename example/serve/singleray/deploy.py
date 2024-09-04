import argparse
from pathlib import Path
import subprocess

MIN_QPS = 5
MAX_QPS = 50
CONFIG = "singleray"
TARGET_QPS = 4


def generate_config(path: Path, cluster_ip):
    with open(path, "w") as f:
        f.write('{\n')
        f.write(f'\t\"http://{cluster_ip}:8000/classifier\": 0.8,\n')
        f.write(f'\t\"http://{cluster_ip}:8000/classifier1\": 0.05,\n')
        f.write(f'\t\"http://{cluster_ip}:8000/classifier2\": 0.05,\n')
        f.write(f'\t\"http://{cluster_ip}:8000/classifier3\": 0.05,\n')
        f.write(f'\t\"http://{cluster_ip}:8000/classifier4\": 0.05\n')
        f.write('}')


def subproc_run(args):
    out = subprocess.run(args, shell=True, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8"))
    result = out.stdout.decode("utf-8").strip()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoscale", action="store_true")
    parser.add_argument("--reverse", action="store_true",
                        help="deploy in the reverse order")
    parser.add_argument("--date", default="04_06")

    args = parser.parse_args()

    script_dir = Path(__file__).parent.absolute()
    work_dir = script_dir.joinpath(
        f"../../../misc/infaas/qps_{MIN_QPS}_{MAX_QPS}")
    deploy_script_path = script_dir.joinpath("../deploy.py")

    cluster_ip = subproc_run(
        "kubectl get services -n k8s-ray | grep serve-cluster1 | awk '{print $3}'")
    print(f"cluster ip: {cluster_ip}")
    assert len(cluster_ip) > 0

    config_path = work_dir.joinpath(f"{CONFIG}_config.json")
    generate_config(config_path, cluster_ip)

    # Deploy replicas
    head_pod = subproc_run(
        "kubectl get pods -n k8s-ray | grep cluster1-ray-head | awk '{print $1}'")
    assert len(head_pod) > 0

    subproc_run(
        f"kubectl cp -n k8s-ray {deploy_script_path} {head_pod}:/home/ray")

    num_replicas_list = [11, 1, 1, 1, 1]
    cmds = []
    for i, num_replicas in enumerate(num_replicas_list):
        min_num_replicas = 2 if i == 0 else 1

        if CONFIG == "fixed":
            assert not args.autoscale
            num_replicas = min_num_replicas

        if args.autoscale:
            num_replicas = num_replicas_list[0]
            cmd_str = "/home/ray/anaconda3/bin/python /home/ray/deploy.py"
            cmd_str += " --autoscale"
            cmd_str += f" --autoscale-target-num={TARGET_QPS}"
            cmd_str += f" --autoscale-min-replicas={min_num_replicas}"
            cmd_str += f" --autoscale-max-replicas={num_replicas}"
            cmd_str += " --serve-dedicated-cpu"
        else:
            cmd_str = "/home/ray/anaconda3/bin/python /home/ray/deploy.py"
            cmd_str += f" --num-replicas={num_replicas}"
            cmd_str += " --serve-dedicated-cpu"

        if i != 0:
            cmd_str += f" --name=classifier{i}"

        cmds.append((head_pod, cmd_str))

    # Run assigner
    assinger_path = script_dir.joinpath("../trace_replayer/assigner.py")
    out_path = work_dir.joinpath(f"{CONFIG}_{args.date}.json")
    if out_path.exists():
        subproc_run(
            f"python {assinger_path} {out_path} --config {config_path} --out {out_path} --reassign")
    else:
        subproc_run(
            f"python {assinger_path} {work_dir.joinpath('twitter_{args.date}_norm.txt')} --config {config_path} --out {out_path}")

    # Copy input
    replayer_pod = "replayer"
    subproc_run(
        f"kubectl cp -n k8s-ray {out_path} {replayer_pod}:/go/input.json")

    if args.reverse:
        cmds = reversed(cmds)

    for head_pod, cmd_str in cmds:
        print(f"[{head_pod}] {cmd_str}")
        subproc_run(
            f"kubectl exec -n k8s-ray --tty pod/{head_pod} -- {cmd_str}")
