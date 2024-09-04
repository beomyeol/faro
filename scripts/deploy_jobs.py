import argparse
import json
import logging
from pathlib import Path
import sys
import subprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--autoscale", action="store_true")
    parser.add_argument("--num-replicas", type=int, default=1,
                        help="num replicas to deploy")
    parser.add_argument("--half", action="store_true",
                        help="make autoscaler parameters half")
    parser.add_argument("--mix", action="store_true",
                        help="use mixed workload")

    args = parser.parse_args()

    format = '%(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.getLevelName(logging.INFO), format=format)

    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir.joinpath("src").as_posix()
    sys.path.append(src_dir)
    import k8s_utils

    script_dir = Path(__file__).parent.resolve()
    root_dir = script_dir.parent

    deploy_script_path = root_dir.joinpath("example").joinpath(
        "serve").joinpath("deploy.py").resolve()
    assert deploy_script_path.exists(), str(deploy_script_path)

    cluster_ips = k8s_utils.list_ray_head_cluster_ips()
    assert len(cluster_ips) > 0
    print("cluster ips:")
    for name, ip in cluster_ips.items():
        print(" ", name, ip)

    # Copy deploy.py
    head_pods = sorted(k8s_utils.list_ray_head_pods().values())
    assert len(head_pods) > 0

    for head_pod in head_pods:
        k8s_utils.copy_to_pod(deploy_script_path, head_pod, "/home/ray")

    cmds = []
    if args.mix:
        print("use mixed workloads", flush=True)
        for i, head_pod in enumerate(head_pods):
            cmd_str = "/home/ray/anaconda3/bin/python /home/ray/deploy.py"
            cmd_str += f" --num-replicas={args.num_replicas}"
            cmd_str += " --serve-dedicated-cpu"
            if i % 2 == 0:
                cmd_str += " --model=resnet18"
            else:
                cmd_str += " --model=resnet34"
            cmd_str += " --max-concurrent-queries=4"
            cmds.append((head_pod, cmd_str))
    else:
        for head_pod in head_pods:
            cmd_str = "/home/ray/anaconda3/bin/python /home/ray/deploy.py"
            cmd_str += f" --num-replicas={args.num_replicas}"
            cmd_str += " --serve-dedicated-cpu"
            cmd_str += " --model=resnet34"
            cmd_str += " --max-concurrent-queries=4"
            cmds.append((head_pod, cmd_str))

    # Generate new input that maps cluster name to ip
    new_lines = []
    input_path = Path(args.input_path)
    for line in input_path.read_text().splitlines():
        obj = json.loads(line)
        urls = {}
        for cluster_name, count_per_job in obj["counts"].items():
            cluster_ip = cluster_ips[cluster_name]
            for job_name, c in count_per_job.items():
                urls[f"http://{cluster_ip}:8000/{job_name}"] = c
        new_obj = {
            "ts": obj["ts"],
            "urls": urls
        }
        new_lines.append(json.dumps(new_obj))
    new_input_path = input_path.with_name("real_input.json")
    print(f"Writing to {new_input_path}", flush=True)
    new_input_path.write_text("\n".join(new_lines))

    # Copy input
    replayer_pod = "replayer"
    k8s_utils.copy_to_pod(new_input_path, replayer_pod, "/go/input.json")

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
