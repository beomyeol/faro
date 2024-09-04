import argparse
from decimal import Decimal
import itertools
import logging
from statistics import mean

import pytorch_lightning as pl
import numpy as np

from configuration import load_config
from resources import Resources
from simulation import metric
from simulation.cluster import ClusterSpec
from simulation.controller import create_controller
from simulation.distribution import create_distribution
from simulation.env import global_env
from simulation.inference import InferenceModelSpec
from simulation.load_generator import create_loadgen
from simulation.simulation import LoadGenCallback
from solver import Solver
from utility import create_utility_func

_LOGGER = logging.getLogger(__name__)


def main(args):
    _LOGGER.info("args: %s", args)
    pl.seed_everything(args.seed)

    config = load_config(args.config)
    latency_config = config["latency_spec"]
    latency_dist = create_distribution(
        latency_config["type"], **latency_config["params"])
    route_latency_config = config["route_latency_spec"]
    route_latency_dist = create_distribution(
        route_latency_config["type"], **route_latency_config["params"])

    processing_time = np.mean([
        latency_dist.draw() + route_latency_dist.draw() + route_latency_dist.draw()
        for _ in range(500)]) * 1e3
    slo_target = 4*processing_time

    num_replicas = args.num_replicas
    if num_replicas is None:
        solver = Solver(
            resource_limit=args.resource_limit,
            adjust=True,
            min_max=False,
            method="cobyla",
            mdc_percentile=args.mdc_percentile,
            upscale_overhead=0,
        )

        util_fn = create_utility_func("latency", slo_target=slo_target)
        solver.add_deployment(
            key="test",
            processing_time=processing_time,
            input_rate=args.input_rate,
            weight=1,
            util_func=util_fn,
            current_num_replicas=1,
            resource_per_replica=1
        )
        sol, _ = solver.solve()
        num_replicas = sol["test"]

    controller = create_controller("default")

    controller.add_node("node", resources=Resources(
        cpu=Decimal(args.resource_limit),
        memory=Decimal(args.resource_limit * 2e9)))

    cluster_spec = ClusterSpec(
        name="cluster",
        quota=None,
        worker_spec=Resources(cpu=Decimal(1), memory=Decimal(1e9)),
        monitor_interval_s=5,
        idle_timeout_s=30,
        route_latency=route_latency_dist,
    )
    controller.deploy_cluster(cluster_spec)

    model_spec = InferenceModelSpec(
        cluster_name="cluster",
        name="model",
        num_replicas=num_replicas,
        resources=Resources(cpu=Decimal(1), memory=Decimal(1e9)),
        latency_dist=latency_dist,
        max_concurrent_queries=4,
        max_queue_len=50
    )
    controller.deploy_model(model_spec)

    callback = LoadGenCallback(controller, 1)
    load_gen = create_loadgen(
        type="poisson", controller=controller, interval_s=1/args.input_rate,
        cluster_name="cluster", job_name="model",
        max_num_tasks=args.max_requests,
    )
    load_gen.run()
    load_gen.proc.callbacks.append(callback.done)

    controller.start()

    global_env().run()

    outputs = {}

    outputs["input_rate"] = args.input_rate
    outputs["mdc"] = args.mdc_percentile
    outputs["processing_time"] = processing_time
    outputs["num_replicas"] = num_replicas
    outputs["latency"] = str(latency_dist)
    outputs["route_latency"] = str(route_latency_dist)

    total_times = np.array(metric.get_total_times()) * 1e3
    # assert metric.num_tasks() == len(total_times)
    outputs["num_requests"] = metric.num_tasks()
    outputs["failed_counts"] = metric.num_tasks() - len(total_times)
    outputs["avg"] = np.mean(total_times)
    outputs["max"] = np.max(total_times)
    count = np.count_nonzero(total_times < slo_target)
    outputs["SLO violation"] = (1-count/metric.num_tasks())*1e2

    df = metric.to_df()
    cluster_names = df.cluster_name.unique()
    latency_dict = {}
    end_tss = []
    for cluster_name in cluster_names:
        cluster_df = df[df.cluster_name == cluster_name]
        tss = cluster_df.arrival_ts.values
        lats = cluster_df.end_ts - cluster_df.arrival_ts
        lats[~cluster_df.succeeded] = np.nan
        latency_dict[cluster_name] = (tss, lats.values * 1e3)
        end_tss.append(cluster_df.end_ts.max())
    max_ts = np.max(end_tss)

    util_dict = {}
    util_fn = create_utility_func("latency", slo_target=slo_target)
    mean_utilities = []
    for cluster_name, (tss, lats) in latency_dict.items():
        utilities = util_fn(lats)
        utilities = np.nan_to_num(utilities, copy=False)
        mean_utility = np.mean(utilities)
        mean_utilities.append(mean_utility)
        util_dict[cluster_name] = (tss, utilities)

    tss, utilities = zip(*util_dict.values())
    outputs["avg_utility"] = mean(utilities)

    # metric.print_stats(target_latency=slo_target/1e3)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-rate", type=int, required=True)
    parser.add_argument("--resource-limit", type=int, default=48)
    parser.add_argument("--mdc-percentile", type=int)
    parser.add_argument("--max-requests", type=int, default=5000)
    parser.add_argument("--num-replicas", type=int)

    format = '%(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.getLevelName(logging.INFO), format=format)

    outputs = main(parser.parse_args())
    metric.print_stats(target_latency=outputs["processing_time"]*4/1e3)
    print(outputs["avg_utility"])
