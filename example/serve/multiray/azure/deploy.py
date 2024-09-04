import argparse
from pathlib import Path
import subprocess
import sys
sys.path.append(str(Path(__file__).absolute().parents[2]))
import k8s_utils

CONFIG = "multiray"
TARGET_QPS = 4


def generate_config(path: Path, cluster_ips):
    print(f"Writing config to {path}")
    with open(path, "w") as f:
        f.write('{')
        for i, (name, ip) in enumerate(cluster_ips.items()):
            f.write("\n" if i == 0 else ",\n")
            fraction = 1 / len(cluster_ips)
            f.write(f'\t\"http://{ip}:8000/classifier\": {fraction}')
        f.write('\n}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k8s", action="store_true")
    parser.add_argument("--autoscale", action="store_true")
    parser.add_argument("--day", type=int, default=7)
    parser.add_argument("--target", required=True, help="target name")
    parser.add_argument("--num-replicas", help="num replicas for manual setup")
    parser.add_argument("--half", action="store_true",
                        help="make autoscaler parameters half")

    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    root_dir = list(script_dir.parents)[3]
    input_dir = root_dir.joinpath("misc").joinpath(
        "azure_2019").joinpath(f"day_{args.day}")
    print("input_dir:", input_dir)
    assert input_dir.exists()
    deploy_script_path = root_dir.joinpath("example").joinpath(
        "serve").joinpath("deploy.py").resolve()
    assert deploy_script_path.exists(), str(deploy_script_path)

    cluster_ips = k8s_utils.get_cluster_ips()
    assert len(cluster_ips) > 0
    print("cluster ips:")
    for name, ip in cluster_ips.items():
        print(" ", name, ip)

    # Copy deploy.py
    head_pods = sorted(k8s_utils.get_head_pods())
    assert len(head_pods) > 0

    for head_pod in head_pods:
        k8s_utils.copy_to_pod(deploy_script_path, head_pod, "/home/ray")

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
                if args.half:
                    cmd_str += " --autoscale-upscale-delay=15"
                    cmd_str += " --autoscale-downscale-delay=300"
                    cmd_str += " --autoscale-metrics-interval=5"
                    cmd_str += " --autoscale-look-back-period=15"
                cmds.append((head_pod, cmd_str))
        else:
            # manual
            assert args.num_replicas
            num_replicas_list = [int(v) for v in args.num_replicas.split(",")]

            for num_replicas, head_pod in zip(num_replicas_list, head_pods):
                cmd_str = "/home/ray/anaconda3/bin/python /home/ray/deploy.py"
                cmd_str += f" --num-replicas={num_replicas}"
                cmd_str += " --serve-dedicated-cpu"
                cmds.append((head_pod, cmd_str))

    # Run assigner
    config_path = input_dir.joinpath(f"{CONFIG.replace('/','_')}_config.json")
    generate_config(config_path, cluster_ips)

    assinger_path = script_dir.joinpath(
        "../../trace_replayer/assigner.py").resolve()
    out_path = input_dir.joinpath(
        f"{CONFIG.replace('/','_')}_{args.target}.json")
    if out_path.exists():
        print(f"Reassigning {out_path}")
        subprocess.check_call(
            ["python", str(assinger_path), str(out_path), "--config",
                str(config_path), "--out", str(out_path), "--reassign"])
    else:
        raise NotImplementedError()

    # Copy input
    replayer_pod = "replayer"
    k8s_utils.copy_to_pod(out_path, replayer_pod, "/go/input.json")

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
