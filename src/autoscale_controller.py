import argparse
import asyncio
import json
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
import time
from typing import Any, Dict, List

import aiohttp
import numpy as np

from autoscale import AutoscaleConfig, create_metric_parser
from autoscaler import Action, AutoscalerExecutor
from configuration import load_config
from k8s_utils import list_ray_head_cluster_ips
from policy import create_policy
from resources import parse_resource_dict


_LOGGER = logging.getLogger(__name__)


class Controller:

    def __init__(self, config: Dict[str, Any], cluster_ips, client):
        self.autoscale_config = AutoscaleConfig.create(**config["autoscale_config"])
        policy_params = config.get("policy_params", {})
        input_rate_bin_size_s = config.get("input_rate_bin_size_s", 1)
        _LOGGER.info("policy: %s, params: %s, config: %s",
                     config["policy_name"], policy_params, self.autoscale_config)
        self.policy = create_policy(
            config["policy_name"], self.autoscale_config, **policy_params,
            input_rate_bin_size_s=input_rate_bin_size_s)

        self._is_working: Dict[str, bool] = {}
        self.action_conns: Dict[str, Connection] = {}
        self.executors: Dict[str, AutoscalerExecutor] = {}
        self._cluster_pods: Dict[str, List[str]] = {}
        self.input_rate_bin_size_s = input_rate_bin_size_s

        self.client = client
        self.cluster_ips = cluster_ips

        self.input_rate_parser = create_metric_parser("input_rate")

        self.worker_pod_resource_dict = {}
        for cluster_spec in config["clusters"]:
            head_name = f"{cluster_spec['name']}-ray-head"
            worker_spec = parse_resource_dict(cluster_spec["worker"])
            self.worker_pod_resource_dict[head_name] = worker_spec

    def generate_executor(self, head_name: str):
        parent_conn, child_conn = mp.Pipe()
        self._is_working[head_name] = False
        self.action_conns[head_name] = parent_conn
        executor = AutoscalerExecutor(
            head_name=head_name,
            autoscale_config=self.autoscale_config,
            conn=child_conn,
            worker_pod_resources=self.worker_pod_resource_dict[head_name])
        executor.start()
        self.executors[head_name] = executor

    async def get_metrics(self):

        async def fetch(session, ip):
            url = f"http://{ip}:8000/-/metrics"
            trials = 0
            while True:
                try:
                    async with session.get(url) as resp:
                        if resp.status != 200:
                            _LOGGER.info("Failed to get input metrics: code=%d",
                                         resp.status)
                            raise RuntimeError()
                        text = await resp.text()
                        await asyncio.sleep(0.0001)
                        return text
                except (aiohttp.ClientError, aiohttp.ServerDisconnectedError) as e:
                    _LOGGER.warning(
                        "aiohttp error (trial=%d): %s", trials + 1, str(e))
                    if trials < 3:
                        trials += 1
                    else:
                        raise e

        results = await asyncio.gather(*[fetch(self.client, cluster_ip)
                                         for cluster_ip in cluster_ips.values()],
                                       return_exceptions=True)
        metrics = {}
        for cluster_name, result in zip(cluster_ips.keys(), results):
            try:
                json_value = json.loads(result)
            except json.JSONDecodeError:
                _LOGGER.info("Failed to parse result str: %s", result)
                json_value = {}
            head_name = cluster_name + "-ray-head"
            metrics[head_name] = json_value
        return metrics

    def _execute_actions(self, actions: List[Action]):
        for action in actions:
            if self.is_working(action.head_name):
                _LOGGER.info("Skip the action since it's still autoscaling %s",
                             action)
                continue
            self._is_working[action.head_name] = True
            self.action_conns[action.head_name].send(action)

    def is_working(self, head_name: str) -> bool:
        if self._is_working[head_name]:
            conn = self.action_conns[head_name]
            if not conn.poll():
                return True
            conn.recv()
            self._is_working[head_name] = False
        return False

    def no_input_check(self, input_metrics):
        total_counts = 0
        for cluster_metrics in input_metrics.values():
            for metrics in cluster_metrics.values():
                current_num_inputs = self.input_rate_parser(metrics)
                if current_num_inputs is not None:
                    total_counts += np.sum(current_num_inputs[0])
        return total_counts == 0

    async def run(self):
        for cluster_name in self.cluster_ips.keys():
            head_name = cluster_name + "-ray-head"
            self.generate_executor(head_name)

        is_started = False
        controller_start_ts = None
        while True:
            start_ts = time.time()
            metrics = await self.get_metrics()
            _LOGGER.info("metrics: %s", metrics)

            if not is_started:
                if self.no_input_check(metrics):
                    # skip if experiment doesn't get started
                    _LOGGER.info("Skip until there is any request")
                    time_to_sleep = self.autoscale_config.interval_s - time.time() + start_ts
                    await asyncio.sleep(time_to_sleep)
                    continue
                else:
                    _LOGGER.info("Started")
                    is_started = True
                    controller_start_ts = start_ts

            metric_end_ts = time.time()
            actions = self.policy.autoscale(
                metrics, metric_end_ts - controller_start_ts)
            _LOGGER.info("autoscale time: %.2f s", time.time() - metric_end_ts)
            self._execute_actions(actions)
            time_to_sleep = self.autoscale_config.interval_s - time.time() + start_ts
            await asyncio.sleep(time_to_sleep)


async def main(config, cluster_ips):
    async with aiohttp.ClientSession() as client:
        controller = Controller(config, cluster_ips, client)
        await controller.run()


if __name__ == "__main__":
    format = '%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=format)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="config file path")
    parser.add_argument("--local", action="store_true",
                        help="use local Ray server")
    parser.add_argument("-n", "--namespace", default="k8s-ray")

    args = parser.parse_args()

    args = parser.parse_args()
    _LOGGER.info("args: %s", args)

    if args.local:
        cluster_ips = {"local": "localhost"}
    else:
        cluster_ips = list_ray_head_cluster_ips(args.namespace)

    _LOGGER.info("cluster_ips: %s", cluster_ips)

    config = load_config(args.config_path)
    new_config = config["controller_params"]
    new_config["clusters"] = config["clusters"]
    _LOGGER.info(str(new_config))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(new_config, cluster_ips))
