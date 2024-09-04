import argparse
from pathlib import Path
import subprocess
import sys

MIN_QPS = 5
MAX_QPS = 50
CONFIG = "multiray/oracle1"
TARGET_QPS = 4


def generate_config(path: Path, cluster_ips):
    print(f"Writing config to {path}")
    with open(path, "w") as f:
        f.write('{')
        for i, (name, ip) in enumerate(cluster_ips):
            f.write("\n" if i == 0 else ",\n")
            fraction = 0.8 if "cluster1" in name else 0.05
            f.write(f'\t\"http://{ip}:8000/classifier\": {fraction}')
        f.write('\n}')


def subproc_run(args):
    out = subprocess.run(args, shell=True, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8"))
    result = out.stdout.decode("utf-8").strip()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k8s", action="store_true")
    parser.add_argument("--autoscale", action="store_true")
    parser.add_argument("--date", default="04_06")
    parser.add_argument("--num-replicas", help="num replicas for manual setup")

    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    root_dir = list(script_dir.parents)[4]
    work_dir = root_dir.joinpath(
        f"misc/infaas/qps_{MIN_QPS}_{MAX_QPS}").resolve()
    work_dir.mkdir(exist_ok=True)
    deploy_script_path = root_dir.joinpath("example").joinpath(
        "serve").joinpath("deploy.py").resolve()
    assert deploy_script_path.exists(), str(deploy_script_path)

    cluster_ips = subproc_run(
        "kubectl get services -n k8s-ray | grep ray-head | awk '{print $1, $3}'")
    cluster_ips = [line.split(" ") for line in cluster_ips.splitlines()]
    assert len(cluster_ips) > 0
    print("cluster ips:")
    for name, ip in cluster_ips:
        print(" ", name, ip)

    config_path = work_dir.joinpath(f"{CONFIG.replace('/','_')}_config.json")
    generate_config(config_path, cluster_ips)

    # Copy deploy.py
    head_pods = subproc_run(
        "kubectl get pods -n k8s-ray | grep ray-head | awk '{print $1}'")
    head_pods = sorted(head_pods.splitlines())
    assert len(head_pods) > 0

    for head_pod in head_pods:
        print(f"Copying {deploy_script_path} to {head_pod}...")
        subproc_run(
            f"kubectl cp -n k8s-ray {deploy_script_path} {head_pod}:/home/ray")

    cmds = []
    num_replicas = 7
    if args.k8s:
        print("Deploying for k8s only autoscaler...")
        for i, head_pod in enumerate(head_pods):
            cmd_str = "/home/ray/anaconda3/bin/python /home/ray/deploy.py"
            cmd_str += f" --num-replicas={num_replicas}"
            cmd_str += " --serve-dedicated-cpu"
            cmds.append((head_pod, cmd_str))
    else:
        if args.autoscale:
            for i, head_pod in enumerate(head_pods):
                min_replicas = 1
                cmd_str = "/home/ray/anaconda3/bin/python /home/ray/deploy.py"
                cmd_str += " --autoscale"
                cmd_str += f" --autoscale-target-num={TARGET_QPS}"
                cmd_str += f" --autoscale-min-replicas={min_replicas}"
                cmd_str += f" --autoscale-max-replicas={num_replicas}"
                cmd_str += " --serve-dedicated-cpu"
                cmds.append((head_pod, cmd_str))
        else:
            if args.num_replicas is not None:
                # manual
                num_replicas_list = [int(v) for v in args.num_replicas.split(",")]
            else:
                # all
                num_replicas_list = [1] * len(head_pods)
                num_replicas_list[0] = 7

            for head_pod, num_replicas in zip(head_pods, num_replicas_list):
                cmd_str = "/home/ray/anaconda3/bin/python /home/ray/deploy.py"
                cmd_str += f" --num-replicas={num_replicas}"
                cmd_str += " --serve-dedicated-cpu"
                cmds.append((head_pod, cmd_str))


    # Run assigner
    assinger_path = root_dir.joinpath(
        "example/serve/trace_replayer/assigner.py").resolve()
    out_path = work_dir.joinpath(f"{CONFIG.replace('/','_')}_{args.date}.json")
    if out_path.exists():
        print(f"Reassigning {out_path}")
        subproc_run(
            f"python {assinger_path} {out_path} --config {config_path} --out {out_path} --reassign")
    else:
        subproc_run(
            f"python {assinger_path} {work_dir.joinpath('twitter_{args.date}_norm.txt')} --config {config_path} --out {out_path}")

    # Copy input
    replayer_pod = "replayer"
    subproc_run(
        f"kubectl cp -n k8s-ray {out_path} {replayer_pod}:/go/input.json")

    # run commands
    # Should use nonblocking subprocess call since the call waits for
    # successful replica deployment, which is prevented by the quota manager
    procs = []
    for head_pod, cmd_str in cmds:
        print(f"[{head_pod}] {cmd_str}")
        args = f"kubectl exec -n k8s-ray --tty pod/{head_pod} -- {cmd_str}"
        proc = subprocess.Popen(args, stdout=sys.stdout,
                                stderr=sys.stderr, shell=True)
        procs.append(proc)

    for proc in procs:
        proc.wait()
