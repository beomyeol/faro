from __future__ import annotations
import argparse
import logging
from typing import Any, Dict

import pytorch_lightning as pl

from simulation.primitives import parse_resource_dict
from simulation.controller import Controller, create_controller
from simulation.cluster import ClusterSpec
from simulation.inference import InferenceModelSpec
from simulation.load_generator import *
from simulation import metric
from simulation.distribution import create_distribution
from simulation.env import global_env
from configuration import load_config


_LOGGER = logging.getLogger(__name__)


class LoadGenCallback():

    def __init__(self, controller: Controller, num_gens: int):
        self.controller = controller
        self.num_gens = num_gens
        self.done_count = 0

    def done(self, _):
        self.done_count += 1
        _LOGGER.info(
            "[%.3f] load gen (%d/%d) is done.",
            self.controller.env.now, self.done_count, self.num_gens)
        if self.done_count == self.num_gens:
            _LOGGER.info("[%.3f] stop controller.",
                         self.controller.env.now)
            self.controller.env.process(self.controller.stop())


def run(config: Dict[str, Any]):
    pl.seed_everything(config.get("seed", None))

    env = global_env()
    controller_params = config.get("controller_params", {})
    controller = create_controller(config["controller"], **controller_params)

    for node_spec in config["nodes"]:
        controller.add_node(
            node_spec["name"],
            parse_resource_dict(node_spec["resources"]))

    for cluster_spec in config["clusters"]:
        quota = parse_resource_dict(
            cluster_spec["quota"]) if "quota" in cluster_spec else None
        spec = ClusterSpec(
            name=cluster_spec["name"],
            quota=quota,
            worker_spec=parse_resource_dict(cluster_spec["worker"]),
            min_workers=(cluster_spec["min_workers"]
                         if "min_workers" in cluster_spec else 1),
        )
        monitor_interval_s = cluster_spec.get("monitor_interval_s", None)
        if monitor_interval_s is not None:
            spec.monitor_interval_s = monitor_interval_s
        idle_timeout_s = cluster_spec.get("idle_timeout_s", None)
        if idle_timeout_s is not None:
            spec.idle_timeout_s = idle_timeout_s
        route_latency_spec = cluster_spec["route_latency"]
        spec.route_latency = create_distribution(
            route_latency_spec["type"],
            **route_latency_spec["params"])
        controller.deploy_cluster(spec)

    for job_spec in config["inference_jobs"]:
        controller.deploy_model(
            InferenceModelSpec(
                cluster_name=job_spec["cluster_name"],
                name=job_spec["name"],
                latency_dist=create_distribution(
                    job_spec["latency_dist"]["type"],
                    **job_spec["latency_dist"]["params"]
                ),
                num_replicas=job_spec["num_replicas"],
                resources=parse_resource_dict(job_spec["resources"]),
                max_concurrent_queries=job_spec.get(
                    "max_concurrent_queries", 100),
                max_queue_len=job_spec.get("max_queue_len", 0),
            )
        )

    callback = LoadGenCallback(controller, len(config["load_specs"]))
    for load_spec in config["load_specs"]:
        load_gen = create_loadgen(
            load_spec["type"], controller, **load_spec["params"])
        load_gen.run()
        load_gen.proc.callbacks.append(callback.done)

    controller.start()

    env.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="yaml config path")
    parser.add_argument("--metric_out_dir", help="metric output dir path")
    parser.add_argument("--metric_dump_path", help="metric pickle dump path")
    parser.add_argument("--target_latency", type=float)
    parser.add_argument("--log_level", choices=logging._nameToLevel,
                        default="INFO")
    parser.add_argument("--stats", action="store_true", help="print stats")
    args = parser.parse_args()

    format = '%(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.getLevelName(args.log_level), format=format)

    config = load_config(args.config_path)
    run(config)

    if args.metric_out_dir:
        metric.write_to_dir(args.metric_out_dir)

    if args.metric_dump_path:
        metric.dump_df(args.metric_dump_path)

    if args.stats:
        target_latency = args.target_latency
        if target_latency is None:
            controller_params = config.get("controller_params", None)
            if controller_params is not None:
                if config["controller"] == "autoscale":
                    autoscale_config_dir = controller_params["autoscale_config"]
                elif config["controller"] == "hybrid":
                    autoscale_config_dir = controller_params["policies"][0]["autoscale_config"]
                target_latency = autoscale_config_dir["target_metric"] / 1e3

        metric.print_stats(target_latency=target_latency)
