import argparse
import subprocess
from pathlib import Path


def subproc_run(args):
    out = subprocess.run(args, shell=True, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.decode("utf-8"))
    result = out.stdout.decode("utf-8").strip()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")

    args = parser.parse_args()

    out_dir = Path(args.outdir).resolve()
    print(f"out_dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    head_pods = subproc_run(
        "kubectl get pods -n k8s-ray -o wide | grep head | awk '{print $1, $6}'")
    head_pods = [line.split(" ") for line in sorted(head_pods.splitlines())]
    print(head_pods)
    assert len(head_pods) > 0

    ip_fpath = out_dir.joinpath("head_ips.log")
    with open(ip_fpath, "w") as f:
        for name, ip in head_pods:
            f.write(f"{name} {ip}\n")

    # copy serve dirs
    serve_dir = out_dir.joinpath("serve")
    serve_dir.mkdir(exist_ok=True)

    for name, _ in head_pods:
        end = name.rfind("-head")
        begin = name.rfind("-", 0, end)
        cluster_name = name[begin+1:end]
        serve_out = serve_dir.joinpath(cluster_name).joinpath("head")
        print(f"copying serve from {name} for {cluster_name}...")
        cmd = "kubectl -n k8s-ray cp {}:/tmp/ray/session_latest/logs/serve {}".format(
            name, serve_out)
        subproc_run(cmd)

    # worker pods
    worker_pods = subproc_run(
        "kubectl get pods -n k8s-ray | grep worker | awk '{print $1}'")
    for name in worker_pods.splitlines():
        end = name.rfind("-worker")
        begin = name.rfind("-", 0, end)
        cluster_name = name[begin+1:end]
        suffix = name[name.rfind("-")+1:]
        serve_out = serve_dir.joinpath(cluster_name).joinpath(f"worker-{suffix}")
        print(f"copying serve from {name} for {cluster_name}...")
        cmd = "kubectl -n k8s-ray cp {}:/tmp/ray/session_latest/logs/serve {}".format(
            name, serve_out)
        try:
            subproc_run(cmd)
        except RuntimeError as e:
            print(e)

    # copy replayer logs
    print("copying logs...")
    subproc_run("kubectl -n k8s-ray cp replayer:/go/start_time.log {}".format(
        out_dir.joinpath("start_time.log")
    ))
    subproc_run("kubectl -n k8s-ray cp replayer:/go/latency.log {}".format(
        out_dir.joinpath("latency.log")
    ))
    subproc_run("kubectl -n k8s-ray cp replayer:/go/out.log {}".format(
        out_dir.joinpath("out.log")
    ))
    subproc_run("kubectl -n k8s-ray cp replayer:/go/run.log {}".format(
        out_dir.joinpath("run.log")
    ))

    try:
        operator_logs = subproc_run(
            "kubectl -n k8s-ray exec deployment/kopf-operator --  cat /run.log")
        operator_log_path = out_dir.joinpath("operator.log")
        operator_log_path.write_text(operator_logs)
    except RuntimeError as e:
        print(e)


if __name__ == "__main__":
    main()
