import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import yaml

_LOGGER = logging.getLogger(__name__)


def assign(sim_input_path, cluster_ips):
    # gen mapping

    out_lines = []

    sim_input_path = Path(sim_input_path)

    for line in sim_input_path.read_text().splitlines():
        obj = json.loads(line)
        urls = {}
        for cluster_name, deps in obj["counts"].items():
            for dep_name, c in deps.items():
                url = f"http://{cluster_ips[cluster_name]}:8000/{dep_name}"
                urls[url] = c

        new_obj = {
            "ts": obj["ts"],
            "urls": urls,
        }
        out_lines.append(json.dumps(new_obj))

    output_path = sim_input_path.with_name("real_input.json")
    output_path.write_text("\n".join(out_lines))

    return output_path.as_posix()


def replace_input_path(config):
    for k in config.keys():
        v = config[k]
        if isinstance(v, dict):
            config[k] = replace_input_path(v)
        elif isinstance(v, str):
            if v.endswith("input.json"):
                new_value = "/home/ray/input.json"
                _LOGGER.info(f"replacing {v} with {new_value}")
                config[k] = new_value
        elif isinstance(v, list):
            config[k] = [
                replace_input_path(item) if isinstance(item, dict) else item
                for item in v
            ]
    return config


def main(args, root_dir: Path):
    autoscale = True
    if args.config_path is not None:
        autoscale = False
        config_path = Path(args.config_path)
        config = load_config(config_path)
        config = replace_input_path(config)
        inference_config_dict = {
            inference_job_config["cluster_name"]: inference_job_config
            for inference_job_config in config["inference_jobs"]
        }

    _LOGGER.info("autoscale: %s", autoscale)

    # copy deploy.py
    deploy_script_path = root_dir.joinpath(
        "example").joinpath("serve").joinpath("deploy.py")
    head_pods = k8s_utils.list_ray_head_pods()
    assert len(head_pods) > 0
    for head_pod in head_pods.values():
        k8s_utils.copy_to_pod(deploy_script_path, head_pod, "/home/ray")

    # generate replica deployment commands
    cmds = []
    for cluster_name, head_pod in head_pods.items():
        cmd_str = "/home/ray/anaconda3/bin/python /home/ray/deploy.py"
        max_concurrent_queries = args.max_concurrent_queries
        if autoscale:
            assert args.autoscale_target_num is not None
            cmd_str += " --autoscale"
            cmd_str += f" --autoscale-target-num={args.autoscale_target_num}"
            cmd_str += f" --autoscale-min-replicas={args.autoscale_min_replicas}"
            cmd_str += f" --autoscale-max-replicas={args.autoscale_max_replicas}"
        else:
            inference_job_config = inference_config_dict[cluster_name]
            cmd_str += f" --num-replicas={inference_job_config['num_replicas']}"
            if "max_concurrent_queries" in inference_job_config:
                max_concurrent_queries = inference_job_config["max_concurrent_queries"]

        cmd_str += " --serve-dedicated-cpu"
        cmd_str += f" --model={args.model}"
        cmd_str += f" --max-concurrent-queries={max_concurrent_queries}"
        cmds.append((head_pod, cmd_str))

    cluster_ips = k8s_utils.list_ray_head_cluster_ips()
    input_path = args.input_path
    assigned_input_path = assign(input_path, cluster_ips)
    _LOGGER.info(f"assigned input path: {assigned_input_path}")

    # copy input to the replayer pod
    k8s_utils.copy_to_pod(assigned_input_path, "replayer", "/go/input.json")

    if not autoscale:
        f = tempfile.NamedTemporaryFile(mode="w", delete=False)
        yaml.safe_dump(config, f)
        f.close()

        # update src files
        controller_pod = k8s_utils.list_autoscale_controller_pod()[0]
        pod_name = controller_pod.metadata.name
        _LOGGER.info(f"controller pod: {pod_name}")
        k8s_utils.exec_pod_cmd(pod_name, "sudo rm -r /home/ray/src")
        k8s_utils.copy_to_pod(
            root_dir.joinpath("src").as_posix(), pod_name, "/home/ray")

        # copy configs to autoscale controller pod
        k8s_utils.copy_to_pod(input_path, pod_name, "/home/ray")
        k8s_utils.copy_to_pod(f.name, pod_name, "/home/ray/config.yaml")
        os.unlink(f.name)

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


if __name__ == "__main__":
    format = '%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=format)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--config_path")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--max-concurrent-queries", type=int, default=100)
    parser.add_argument("--autoscale-target-num", type=int,
                        help="autoscaling target num ongoing requests")
    parser.add_argument("--autoscale-min-replicas", type=int, default=1)
    parser.add_argument("--autoscale-max-replicas", type=int, default=20)
    parser.add_argument("--autoscale-upscale-delay", type=float)
    parser.add_argument("--autoscale-downscale-delay", type=float)
    parser.add_argument("--autoscale-metrics-interval", type=float)
    parser.add_argument("--autoscale-look-back-period", type=float)

    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir.joinpath("src").as_posix()
    sys.path.append(src_dir)
    from configuration import load_config
    import k8s_utils

    main(parser.parse_args(), root_dir)
