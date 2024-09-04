import logging
import yaml
import tempfile
import argparse
from pathlib import Path
import subprocess
import sys


def get_operator_pod():
    pods = subprocess.check_output(
        ["kubectl", "get", "pods", "-n", "k8s-ray"]).decode("utf-8")
    for line in pods.splitlines():
        if "faro-operator" in line:
            split = line.split()
            return split[0]
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")

    args = parser.parse_args()

    format = '%(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.getLevelName(logging.INFO), format=format)

    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir.joinpath("src").as_posix()
    sys.path.append(src_dir)
    from configuration import load_config
    import k8s_utils

    config = load_config(args.config_path)
    pod_name = get_operator_pod()

    with tempfile.NamedTemporaryFile("w") as f:
        yaml.dump(config, f)
        f.flush()
        k8s_utils.copy_to_pod(f.name, pod_name, "/home/ray/config.yaml")
