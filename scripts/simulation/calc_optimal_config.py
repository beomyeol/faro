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

    src_path = Path(__file__).parent.joinpath("..").joinpath("..").joinpath("src")
    sys.path.append(src_path.as_posix())

    from configuration import load_config
    from policy import create_policy
    from autoscale import AutoscaleConfig, QUEUE_LEN_KEY, INPUT_RATE_KEY, HEAD_QUEUE_LEN_KEY

    config_path = Path(args.config_path).resolve()
    config = load_config(config_path)
    policy_params = config.get("policy_params", {})
    autoscale_config = AutoscaleConfig(**config["autoscale_config"])
    _LOGGER.info("%s", autoscale_config)
    input_rate_bin_size_s = config["input_rate_bin_size_s"]

    os.chdir(src_path)
    policy = create_policy(config["policy_name"], autoscale_config, **policy_params, input_rate_bin_size_s=input_rate_bin_size_s)

    metrics = {}

    for job in config["inference_jobs"]:
        head_name = job["cluster_name"] + "-ray-head"
        metrics[head_name] = {job["name"]: {
            QUEUE_LEN_KEY: [0],
            INPUT_RATE_KEY: ([1], 1),
            HEAD_QUEUE_LEN_KEY: 0,
        }}

    for preds in policy.predictors.values():
        for pred in preds.values():
            pred.non_zero_idx -= 1

    actions = policy.autoscale(metrics, current_ts=0)
    num_replicas = {action.head_name: {action.deployment_name: action.target} for action in actions}

    for job in config["inference_jobs"]:
        head_name = job["cluster_name"] + "-ray-head"
        job["num_replicas"] = int(num_replicas[head_name][job["name"]])

    out_path = config_path.with_name("job_specs.yaml")

    with open(out_path, "w") as f:
        yaml.dump(config["inference_jobs"], f)