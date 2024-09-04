from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from math import ceil
from threading import Thread
import multiprocessing as mp
from multiprocessing.connection import Connection
import logging
import time
import asyncio
from typing import List, Dict, Optional, Tuple
import argparse
import json
import os

import aiohttp
import numpy as np
import ray
import ray.serve
import requests
from autoscale import Action, AutoscaleConfig
from configuration import load_config

from k8s_utils import (
    list_ray_head_cluster_ips,
    get_nodes_free_resources,
    parse_quantity,
    get_ray_cluster_pods,
    get_available_resource_quota,
)
from policy import create_policy
from ray.serve._private.custom_autoscaling import (
    INPUT_RATE_KEY, QUEUE_LEN_KEY, DROP_RATE_KEY,
)
from resources import Resources, parse_resource_dict

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)
_IN_KUBERNETES_POD = "KUBERNETES_SERVICE_HOST" in os.environ


def calculate_num_replicas(action: Action, current_replicas: int,
                           min_replicas: int, max_replicas: int):
    if action.factor is not None:
        assert action.delta is None
        delta = ceil(current_replicas * action.factor) - current_replicas
    elif action.target is not None:
        delta = action.target - current_replicas
    else:
        delta = action.delta
    return min(max(min_replicas, current_replicas + delta),
               max_replicas)


@dataclass
class State:
    num_replicas: Dict[str, Dict[str, int]] = None
    available_quota: Resources = None
    autoscale_config: AutoscaleConfig = None
    worker_pod_resource_dict: Dict[str, Resources] = None


def check_action(state: State, action: Action) -> Tuple[Optional[Action], Optional[Action]]:
    current_num_replicas = state.num_replicas[action.head_name][action.deployment_name]
    _LOGGER.debug("Check action: %s, current_replicas: %d",
                  action, current_num_replicas)
    new_num_replicas = calculate_num_replicas(
        action, current_num_replicas,
        min_replicas=state.autoscale_config.min_replicas,
        max_replicas=state.autoscale_config.max_replicas)
    delta = new_num_replicas - current_num_replicas

    new_action, remainder = None, None
    if delta > 0:
        # TODO: assume that available resources >= quota.
        # TODO: only check cpu for now
        # upscale. check quota
        if state.available_quota is not None:
            remained_cpu = state.available_quota.cpu - delta * \
                state.worker_pod_resource_dict[action.head_name].cpu
            _LOGGER.debug(
                "avail: %d, delta: %d, remained: %d",
                state.available_quota.cpu, delta, remained_cpu)
            if remained_cpu < 0:
                # not all delta can be deployed.
                delta += remained_cpu
                if action.trials > 1:
                    remainder = Action(
                        head_name=action.head_name,
                        deployment_name=action.deployment_name,
                        delta=int(-remained_cpu),
                        accept_rate=action.accept_rate,
                        trials=action.trials - 1)
                    _LOGGER.debug("Remainder: %s", remainder)
    if delta != 0 or action.accept_rate is not None:
        # need to update accept rate if it exists.
        # TODO: skip if the new accept rate is the same as the current one

        # delta can be Decimal.
        new_num_replicas = int(current_num_replicas + delta)
        new_action = Action(head_name=action.head_name,
                            deployment_name=action.deployment_name,
                            target=new_num_replicas,
                            accept_rate=action.accept_rate)
        # update status
        state.num_replicas[action.head_name][action.deployment_name] = new_num_replicas
        if state.available_quota is not None:
            state.available_quota.cpu -= delta * \
                state.worker_pod_resource_dict[action.head_name].cpu
            _LOGGER.debug("New action: %s. avail: %d",
                          new_action, state.available_quota.cpu)
        else:
            _LOGGER.debug("New action: %s", new_action)

    return new_action, remainder


# class AutoscaleCalculator:

#     def __init__(self, head_name: str, min_replicas: int, max_replicas: int,
#                  worker_pod_resources: Resources):
#         self.head_name = head_name
#         self.min_replicas = min_replicas
#         self.max_replicas = max_replicas
#         self.worker_pod_resources = worker_pod_resources

#         _LOGGER.info("head_name=%s, worker_resources=%s", head_name,
#                      str(worker_pod_resources))

#     def try_allocation(self, num_requests: int, resource_request: Resources):

#         def _try_allocation(free_resources, num, unit_resource):
#             left_num = num
#             for avail in free_resources.values():
#                 while left_num > 0 and avail >= unit_resource:
#                     avail -= unit_resource
#                     left_num -= 1
#             return left_num

#         if num_requests <= 0:
#             raise ValueError(
#                 f"num_requests ({num_requests}) should be > 0")

#         # tried to use ray._private.state.state._available_resources_per_node()
#         # but it failed by saying that ray.init() is needed although it's done
#         free_ray_node_resources = {
#             "cluster": Resources(
#                 cpu=Decimal(ray.available_resources().get("CPU", 0)),
#                 memory=Decimal(0))
#         }
#         # only consider CPU for now.
#         # TODO: consider memory. how to know how much memory is
#         # required for each replicas?
#         left_num_requests = _try_allocation(
#             free_ray_node_resources, num_requests, resource_request)

#         # consider free space in a k8s cluster
#         # TODO: it is assumed that only a single replica fits into a worker pod
#         #       this should change if the assumption does not hold
#         if left_num_requests > 0 and _IN_KUBERNETES_POD:
#             free_node_resources = get_nodes_free_resources()
#             left_num_requests = _try_allocation(
#                 free_node_resources, left_num_requests,
#                 self.worker_pod_resources)

#         return left_num_requests

#     def __call__(self, action: Action, dep):
#         if action.factor is not None:
#             assert action.delta is None
#             delta = ceil(dep.num_replicas * action.factor) - dep.num_replicas
#         elif action.target is not None:
#             delta = action.target - dep.num_replicas
#         else:
#             delta = action.delta
#         new_num_replicas = min(
#             max(self.min_replicas, dep.num_replicas + delta),
#             self.max_replicas)
#         delta = new_num_replicas - dep.num_replicas
#         if action.limit is not None and abs(delta) > action.limit:
#             old_delta = delta
#             delta = action.limit * (1 if old_delta > 0 else -1)
#             _LOGGER.info("Limiting delta: from %d to %d", old_delta, delta)
#             new_num_replicas = dep.num_replicas + delta
#         if delta > 0:
#             # upscale
#             left_delta = self.try_allocation(delta, action.resources)
#             if left_delta > 0:
#                 _LOGGER.warn(
#                     "No available resources to deploy %d replicas for [%s]%s. "
#                     "requested=%d",
#                     left_delta, self.head_name, action.deployment_name, delta)
#                 # TODO: this may be fixed in real implementation?
#                 # new_num_replicas -= left_delta
#         return new_num_replicas


class AutoscalerExecutor(mp.Process):

    def __init__(self, head_name: str, autoscale_config: AutoscaleConfig,
                 conn: Connection, worker_pod_resources: Resources,
                 head_url: str):
        super().__init__()
        self.head_name = head_name
        self.autoscale_config = autoscale_config
        self._conn = conn
        # self.calculator = AutoscaleCalculator(
        #     head_name=self.head_name,
        #     min_replicas=self.autoscale_config.min_replicas,
        #     max_replicas=self.autoscale_config.max_replicas,
        #     worker_pod_resources=worker_pod_resources,
        # )
        self.worker_pod_resources = worker_pod_resources
        self.head_url = head_url

    def run(self):
        _LOGGER.info("Starting executor for %s...", self.head_name)
        # pregenerate clients since each takes about 5-10 seconds
        self.init_ray(self.head_name)
        http_session = requests.Session()
        while True:
            action: Action = self._conn.recv()
            if action is None:
                break
            dep = ray.serve.get_deployment(action.deployment_name)
            # new_num_replicas = self.calculator(action, dep)

            # Only target is supported for now by the executor.
            # The action should be proprocssed by check_action()
            new_num_replicas = action.target

            # Currently this method is a blocking.
            # Calling deploy(_blocking=False) can make this unblocking, but
            # this causes the process does not complete well
            if dep.num_replicas == new_num_replicas:
                _LOGGER.info("No num_replica changes for [%s]%s: %d",
                             self.head_name, action.deployment_name,
                             new_num_replicas)
            else:
                _LOGGER.info("Changing num_replicas for [%s]%s from %d to %d",
                             self.head_name,
                             action.deployment_name,
                             dep.num_replicas,
                             new_num_replicas)
                start_ts = time.perf_counter()
                if new_num_replicas > dep.num_replicas:
                    # upscale
                    # need to wait for autoscaler update to avoid overallocation
                    # without this, Ray autoscaler launches new worker nodes,
                    # assigns replicas for new CPUs, *but* does not change
                    # resource demand in time, which leads to allocate resources
                    # again....
                    request_cpu = ceil(
                        new_num_replicas * self.worker_pod_resources.cpu + 1)  # 1 for head
                    ray.autoscaler.sdk.request_resources(num_cpus=request_cpu)
                    while ray.cluster_resources().get('CPU', 0) < request_cpu:
                        time.sleep(0.1)
                    time.sleep(6)  # 5s is autoscaler update interval
                    dep.options(num_replicas=new_num_replicas).deploy()
                    # reset for downscaling later
                    ray.autoscaler.sdk.request_resources(num_cpus=0)
                else:
                    dep.options(num_replicas=new_num_replicas).deploy()
                _LOGGER.info(
                    "Changing replica for [%s]%s is done. elapsed time: %.2f s",
                    self.head_name, action.deployment_name,
                    time.perf_counter() - start_ts)
            # send a done msg
            self._conn.send((action.deployment_name, new_num_replicas))
            # send a accept rate request to the head if it exists.
            if action.accept_rate is not None:
                trials = 0
                while trials < 3:
                    trials += 1
                    _LOGGER.info("Sending accept rate for %s: %.4f",
                                 self.head_name, action.accept_rate)
                    r = http_session.post(
                        self.head_url, json.dumps({"rate": action.accept_rate}))
                    if r.status_code == 200:
                        break
                    else:
                        _LOGGER.info(
                            "Failed to send accept rate for %s: %.4f. status=%d trial=%d",
                            self.head_name, action.accept_rate, r.status_code, trials)
                        time.sleep(1.0)
        _LOGGER.info("Stopping executor for %s...", self.head_name)

    @staticmethod
    def init_ray(head_name: str):
        # try to generate new cli
        _LOGGER.info("Creating a new cli for %s", head_name)
        if head_name == "local-ray-head":
            ray.init(address="auto")
        else:
            ray.init(f"ray://{head_name}:10001")
        _LOGGER.info("New cli for %s is done", head_name)


class Autoscaler(Thread):

    def __init__(self,
                 namespace: str,
                 config: Dict,
                 use_local_cluster=False,
                 vcpu_factor=1):
        super().__init__()
        self.namespace = namespace
        self.autoscale_config = AutoscaleConfig.create(
            **config["autoscale_config"])
        self._should_stop = False
        self._use_local_cluster = use_local_cluster

        policy_params = config.get("policy_params", {})
        self.unit_time = config.get("unit_time", 1)
        self.policy = create_policy(
            config["policy_name"], self.autoscale_config, **policy_params,
            input_rate_bin_size_s=self.unit_time)

        self._is_working: Dict[str, bool] = {}
        self.action_conns: Dict[str, Connection] = {}
        self.executors: Dict[str, AutoscalerExecutor] = {}
        self._cluster_pods: Dict[str, List[str]] = {}
        self._cluster_ips = self._get_cluster_ips()

        self.worker_pod_resource_dict = {}
        for cluster_spec in config["clusters"]:
            head_name = f"{cluster_spec['name']}-ray-head"
            worker_spec = parse_resource_dict(cluster_spec["worker"])
            self.worker_pod_resource_dict[head_name] = worker_spec

        self.state = State(
            num_replicas=defaultdict(dict),
            available_quota=(get_available_resource_quota(factor=vcpu_factor)
                             if _IN_KUBERNETES_POD else None),
            autoscale_config=self.autoscale_config,
            worker_pod_resource_dict=self.worker_pod_resource_dict,
        )
        _LOGGER.info("Available Quota: %s", self.state.available_quota)

    @property
    def num_replicas(self):
        return self.state.num_replicas

    def generate_executor(self, head_name: str):
        parent_conn, child_conn = mp.Pipe()
        self._is_working[head_name] = False
        self.action_conns[head_name] = parent_conn
        cluster_name = head_name.replace("-ray-head", "")
        head_url = f"http://{self._cluster_ips[cluster_name]}:8000/-/accept_rate"
        executor = AutoscalerExecutor(
            head_name=head_name,
            autoscale_config=self.autoscale_config,
            conn=child_conn,
            worker_pod_resources=self.worker_pod_resource_dict[head_name],
            head_url=head_url)
        executor.start()
        self.executors[head_name] = executor

    def stop(self):
        self._should_stop = True

    @staticmethod
    def sort_actions_by_delta(
            actions: List[Action],
            num_replicas: Dict[str, Dict[str, int]],
            min_replicas: int,
            max_replicas: int,
            only_downscale_first: bool = False,
    ) -> List[Action]:
        """Sort actions by delta.

        if only_downscale_first is True, only downscale actions are placed upfront.
        The order among the downscale actions and the order among the others remain.
        """
        actions_with_delta = []
        for action in actions:
            current_replicas = num_replicas[action.head_name][action.deployment_name]
            new_num_replicas = calculate_num_replicas(
                action, current_replicas, min_replicas, max_replicas)
            delta = new_num_replicas - current_replicas
            if only_downscale_first:
                # False (downscale) comes before True (others)
                delta = (delta >= 0)
            actions_with_delta.append((delta, action))
        return [action for _, action
                in sorted(actions_with_delta, key=lambda x: x[0])]

    def _execute_actions(self, actions: List[Action]):
        remained_actions = []
        # should process downscale first to let upscale get resources.
        # Is there a case where upscale and downscale for the same deployment
        # exist and the upscale is ahead of the downscale in the actions?
        # This can happen only when the upscale action with high trial counts
        # existed before and a new downscale action comes.
        # Actions with high trial counts are generated only with utility policies.
        # If the utility policy interval > high trial counts * self.interval,
        # the upscale action cannot be ahead of the downscale action.
        actions = self.sort_actions_by_delta(
            actions, self.num_replicas, self.autoscale_config.min_replicas,
            self.autoscale_config.max_replicas, only_downscale_first=True)
        for action in actions:
            if self.is_working(action.head_name):
                _LOGGER.info(
                    "Skip the action since it's still autoscaling. %s",
                    action)
                action.trials -= 1
                if action.trials > 0:
                    remained_actions.append(action)
                continue

            action, remainer = check_action(self.state, action)
            if remainer is not None:
                remained_actions.append(remainer)
            if action is not None:
                self._is_working[action.head_name] = True
                self.action_conns[action.head_name].send(action)
        return remained_actions

    def is_working(self, head_name: str) -> bool:
        if self._is_working[head_name]:
            conn = self.action_conns[head_name]
            if not conn.poll():
                return True
            deployment_name, new_num_replicas = conn.recv()
            assert self.num_replicas[head_name][deployment_name] == new_num_replicas, \
                f"{self.num_replicas[head_name][deployment_name]} != {new_num_replicas}"
            self._is_working[head_name] = False
        return False

    def _get_metrics_for_not_working_clusters(self, metrics: Dict[str, dict]):
        # check whether autoscale actions are done
        return {
            head_name: cluster_metrics
            for head_name, cluster_metrics in metrics.items()
            if not self.is_working(head_name)
        }

    def _log_running_ray_pods(self):
        if not _IN_KUBERNETES_POD:
            return
        field_selector = ",".join(
            [
                "status.phase!=Failed",
                "status.phase!=Unknown",
                "status.phase!=Succeeded",
                "status.phase!=Terminating",
                "status.phase!=Pending",
            ]
        )
        cluster_pods = defaultdict(list)
        pods = get_ray_cluster_pods(namespace=self.namespace,
                                    field_selector=field_selector)
        for pod in pods:
            cluster_name = pod.metadata.labels["ray-cluster-name"]
            cluster_pods[cluster_name].append(pod.metadata.name)
        for cluster_name, pod_names in cluster_pods.items():
            pod_names = sorted(pod_names)
            if cluster_name in self._cluster_pods:
                if self._cluster_pods[cluster_name] == pod_names:
                    # no change
                    continue
            self._cluster_pods[cluster_name] = pod_names
            _LOGGER.info("Running pods. cluster_name=%s, pods=%s",
                         cluster_name, pod_names)

    def initialize(self):
        for cluster_name in self._cluster_ips.keys():
            head_name = f"{cluster_name}-ray-head"
            self.generate_executor(head_name)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # initialize num replicas
        metrics = self._get_autoscaling_metrics()
        for head_name, cluster_metrics in metrics.items():
            for job_name, job_metrics in cluster_metrics.items():
                self.num_replicas[head_name][job_name] = len(
                    job_metrics[QUEUE_LEN_KEY])

    def run(self):
        _LOGGER.info("Starting autoscaler")
        self.initialize()

        if "utility" in self.policy.__class__.__name__.lower():
            raise ValueError(
                "Utility class is not supported in the normal autoscaler")

        remained_actions = []
        base_ts = time.perf_counter()
        while not self._should_stop:
            ts = time.perf_counter()
            self._log_running_ray_pods()
            metrics = self._get_autoscaling_metrics()
            _LOGGER.info("Metrics: %s", metrics)
            # TODO: this will not work well for utility policies.
            metrics = self._get_metrics_for_not_working_clusters(metrics)
            start_ts = time.perf_counter()
            actions = self.policy.autoscale(metrics, start_ts - base_ts)
            autoscale_time = time.perf_counter() - start_ts
            _LOGGER.info("%s autoscale time: %.3f",
                         self.policy.__class__.__name__, autoscale_time)
            remained_actions.extend(actions)
            remained_actions = self._execute_actions(remained_actions)
            time_to_sleep = self.autoscale_config.interval_s - time.perf_counter() + ts
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            else:
                _LOGGER.warning(
                    "Skip to sleep. time_to_sleep=%s s", time_to_sleep)
        _LOGGER.info("Stopping autoscaler")
        for action_connn in self.action_conns.values():
            action_connn.send(None)
        for executor in self.executors.values():
            executor.join()

    def _get_cluster_ips(self):
        if self._use_local_cluster:
            return {"local": "127.0.0.1"}
        else:
            return list_ray_head_cluster_ips(self.namespace)

    def _get_autoscaling_metrics(self):
        async def fetch(session, ip):
            url = f"http://{ip}:8000/-/metrics"
            try:
                async with session.get(url) as response:
                    return await response.text()
            except aiohttp.ClientError as e:
                return str(e)

        async def send_requests(cluster_ips):
            # reusing session causes an error...
            async with aiohttp.ClientSession() as session:
                return await asyncio.gather(
                    *[fetch(session, cluster_ip)
                      for cluster_ip in cluster_ips.values()])

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(send_requests(self._cluster_ips))
        metrics = {}
        for cluster_name, result in zip(self._cluster_ips.keys(), results):
            try:
                json_value = json.loads(result)
            except json.JSONDecodeError:
                _LOGGER.info("Failed to parse result str: %s", result)
                json_value = {}
            head_name = cluster_name + "-ray-head"
            metrics[head_name] = json_value

        return metrics


class HybridAutoscaler(Autoscaler):

    def __init__(self,
                 namespace: str,
                 config: Dict,
                 use_local_cluster=False,
                 vcpu_factor=1):
        # generate a config for a single policy
        single_policy_config = deepcopy(config)
        single_policy_config.update(config["policies"][0])
        super().__init__(
            namespace, single_policy_config, use_local_cluster, vcpu_factor)

        input_rate_bin_size_s = config.get("unit_time", 1)
        self.policies = [self.policy]
        for policy_config in config["policies"][1:]:
            policy_name = policy_config["policy_name"]
            autoscale_config = AutoscaleConfig.create(
                **policy_config["autoscale_config"])
            policy_params = policy_config.get("policy_params", {})
            self.policies.append(
                create_policy(policy_name, autoscale_config, **policy_params,
                              input_rate_bin_size_s=input_rate_bin_size_s))

        interval_s_list = [
            policy.autoscale_config.interval_s for policy in self.policies
        ]
        self.interval_s = min(interval_s_list)
        _LOGGER.info("interval_s: %d", self.interval_s)
        self.interval_counts = []
        for interval_s in interval_s_list:
            if interval_s % self.interval_s != 0:
                raise ValueError(f"incompatible interval_s: {interval_s}")
            self.interval_counts.append(interval_s / self.interval_s)

        self.input_rates = defaultdict(lambda: defaultdict(list))

    @staticmethod
    def merge_input_rates(input_rates):
        first_req_ts = None
        prev_ts = None
        history = []
        for ts, input_metrics in input_rates:
            if first_req_ts is None:
                prev_ts = ts
                if input_metrics is not None:
                    counts, req_ts = input_metrics
                    first_req_ts = prev_ts - req_ts
                    history = np.array(counts, dtype=np.int32)
            else:
                history = np.pad(history, (0, int(round(ts - prev_ts, 8))),
                                 constant_values=0)
                prev_ts = ts
                if input_metrics is not None:
                    new_counts, max_ts = input_metrics
                    history[-min(ceil(max_ts), len(new_counts)):] += new_counts

        if first_req_ts is None:
            return None
        else:
            return [history, prev_ts - first_req_ts]

    @staticmethod
    def aggregate_counts(counts, unit):
        remainder = len(counts) % unit
        size = int(len(counts) / unit)
        aggregated = []
        if remainder > 0:
            aggregated.append(np.sum(counts[:remainder]))
        if size > 0:
            aggregated.extend([np.sum(split) for split
                               in np.split(np.array(counts[remainder:]), size)])
        return aggregated

    def _get_policy_autoscaling_metrics(self, policy_idx):
        metrics = self._get_autoscaling_metrics()
        _LOGGER.info("Raw metrics: %s", metrics)
        current_ts = time.perf_counter()
        # aggregate input rate
        for head_name, cluster_metrics in metrics.items():
            for job_name, job_metrics in cluster_metrics.items():
                input_rates = self.input_rates[head_name][job_name]
                input_rates.append(
                    (current_ts, job_metrics.get(INPUT_RATE_KEY, None)))
                # merge input rates into a single one
                merged = self.merge_input_rates(input_rates)
                if merged is not None:
                    counts, max_ts = merged
                    job_metrics[INPUT_RATE_KEY] = [
                        self.aggregate_counts(
                            counts, self.unit_time),
                        max_ts]
                if policy_idx == 0:
                    input_rates.clear()
        return metrics

    def run(self):
        _LOGGER.info("Starting hybrid autoscaler")
        self.initialize()

        current_count = 0
        remained_actions = []
        base_ts = time.perf_counter()
        while not self._should_stop:
            current_count += 1
            ts = time.perf_counter()
            self._log_running_ray_pods()

            for i, (interval_count, policy) in enumerate(zip(self.interval_counts, self.policies)):
                run_autoscale = current_count % interval_count == 0
                if run_autoscale:
                    metrics = self._get_policy_autoscaling_metrics(i)
                    if i > 0:
                        # assume that only 0-th policy is the utility policy
                        # disable autoscaling while the autoscaling action is in progress
                        metrics = self._get_metrics_for_not_working_clusters(
                            metrics)
                    _LOGGER.info("Metrics: %s", metrics)
                    start_ts = time.perf_counter()
                    actions = policy.autoscale(metrics, start_ts - base_ts)
                    if i == 0:
                        # highest level actions have more trials
                        for action in actions:
                            action.trials = 5
                    autoscale_time = time.perf_counter() - start_ts
                    _LOGGER.info("%s autoscale time: %.3f",
                                 policy.__class__.__name__, autoscale_time)
                    break
            for policy in self.policies[i+1:]:
                # clear policies with lower priorities
                policy.clear()

            remained_actions.extend(actions)
            remained_actions = self._execute_actions(remained_actions)
            time_to_sleep = self.interval_s - time.perf_counter() + ts
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            else:
                _LOGGER.warning(
                    "Skip to sleep. time_to_sleep=%s s", time_to_sleep)

        _LOGGER.info("Stopping autoscaler")
        for action_connn in self.action_conns.values():
            action_connn.send(None)
        for executor in self.executors.values():
            executor.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="config file path")
    parser.add_argument("--local", action="store_true",
                        help="use local Ray server")
    parser.add_argument("-n", "--namespace", default="k8s-ray")
    parser.add_argument("--log_level", choices=logging._nameToLevel,
                        default="INFO")
    parser.add_argument("--vcpu_factor", type=float, default=1.0)

    args = parser.parse_args()

    format = '[%(asctime)s] %(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.getLevelName(
        args.log_level), format=format)

    config = load_config(args.config_path)
    _LOGGER.info(str(config))
    if config["policy_name"] == "hybrid":
        autoscaler = HybridAutoscaler(namespace=args.namespace,
                                      config=config,
                                      use_local_cluster=args.local,
                                      vcpu_factor=args.vcpu_factor)
    else:
        autoscaler = Autoscaler(namespace=args.namespace,
                                config=config,
                                use_local_cluster=args.local,
                                vcpu_factor=args.vcpu_factor)
    autoscaler.start()
    try:
        autoscaler.join()
    except KeyboardInterrupt:
        autoscaler.stop()
        autoscaler.join()
