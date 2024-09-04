import logging
import os
import argparse
from pathlib import Path
import sys
import yaml

_LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")

    args = parser.parse_args()

    format = '%(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.getLevelName(logging.DEBUG), format=format)

    src_path = Path(__file__).parent.joinpath(
        "..").joinpath("..").joinpath("src")
    sys.path.append(src_path.as_posix())

    from configuration import load_config

    config_path = Path(args.config_path).resolve()
    config = load_config(config_path)

    total_cpus = 0
    for node_spec in config["nodes"]:
        total_cpus += node_spec["resources"]["cpu"]

    num_jobs = len(config["inference_jobs"])
    num_replicas = int(total_cpus * 0.8 / num_jobs)
    _LOGGER.info("total_cpus: %d, #jobs: %d #replicas: %d",
                 total_cpus, num_jobs, num_replicas)

    for job in config["inference_jobs"]:
        job["num_replicas"] = num_replicas

    out_path = config_path.with_name("job_specs.yaml")

    with open(out_path, "w") as f:
        yaml.dump(config["inference_jobs"], f)
