from collections import defaultdict
import logging
from math import ceil
import time
from typing import Any, Dict, List, Optional
from copy import deepcopy

import simpy
import numpy as np
from ray.serve._private.custom_autoscaling import (
    INPUT_RATE_KEY,
    LATENCY_KEY,
    HEAD_QUEUE_LEN_KEY,
    QUEUE_LEN_KEY,
    DROP_RATE_KEY,
    TimeMetricStore,
)

from simulation import metric
from simulation.cluster import Cluster, ClusterSpec
from simulation.primitives import Node, Resources, Pod, Task
from simulation.inference import InferenceModelSpec, InferenceTask
from simulation.env import global_env
from autoscale import Action, AutoscaleConfig
from autoscaler import Autoscaler, State, check_action
from policy import Policy, create_policy

_LOGGER = logging.getLogger(__name__)


class Controller:

    def __init__(self, **kwargs) -> None:
        self.env = global_env()
        self.proc: simpy.Process = None
        self.clusters: Dict[str, Cluster] = {}
        self.cluster_quotas: Dict[str, Resources] = {}

        self.nodes: Dict[str, Node] = {}
        self.params = kwargs

    def start(self):
        self.proc = self.env.process(self.run())

    def estimate_num_allocatable_units(self, unit_resource: Resources):
        num_units = 0
        for node in self.nodes.values():
            avail = node.available_resources
            while avail >= unit_resource:
                avail -= unit_resource
                num_units += 1
        return num_units

    def available_resources(self) -> Dict[str, Resources]:
        return {
            node.name: node.available_resources
            for node in self.nodes.values()
        }

    def try_allocation(self, cluster_name: str, num_requests: int,
                       resource_request: Resources) -> int:

        def _try_allocation(free_resources, num, unit_resource):
            free_resources = deepcopy(free_resources)
            left_num = num
            for avail in free_resources.values():
                while left_num > 0 and avail >= unit_resource:
                    avail -= unit_resource
                    left_num -= 1
            return left_num

        if num_requests <= 0:
            raise ValueError(f"num_request ({num_requests}) should be > 0")

        cluster = self.clusters[cluster_name]
        available_resources = {f"worker{i}": res for i, res
                               in enumerate(cluster.available_resources())}

        left_num_request = _try_allocation(
            available_resources, num_requests, resource_request)

        # consider free space in the cluster
        if left_num_request > 0:
            allocatable_num_pods = self.estimate_num_allocatable_units(
                cluster.spec.worker_spec)
            allocatable_worker_resources = {
                f"worker{i}": deepcopy(cluster.spec.worker_spec)
                for i in range(allocatable_num_pods)
            }
            left_num_request = _try_allocation(
                allocatable_worker_resources, left_num_request,
                resource_request)

        return left_num_request

    def add_node(self, name: str, resources: Resources):
        self.nodes[name] = Node(name, resources)

    def log(self, msg, *args):
        new_msg = "[%.3f] " + msg
        new_args = (self.env.now, *args)
        return (new_msg, *new_args)

    def check_quota(self, cluster_name: str, request: Resources) -> bool:
        cluster = self.clusters[cluster_name]
        quota = self.cluster_quotas.get(cluster_name, None)
        if quota is None:
            # no quota for this cluster. admit this for now
            return True

        used = cluster.used_resources()

        if used + request <= quota:
            return True

        _LOGGER.info(*self.log(
            "pod creation request for %s is denied: quota exceeded. "
            "request=%s, used=%s, quota=%s",
            cluster_name, request, used, quota))
        return False

    def create_pod(self, request: Resources, cluster_name: str) -> Optional[Pod]:
        # TODO: admission controller
        if not self.check_quota(cluster_name, request):
            return None

        for node in self.nodes.values():
            if node.available_resources >= request:
                pod = node.create_pod(request)
                _LOGGER.debug(*self.log("created %s. avail=%s/%s",
                              pod, node.available_resources,
                              node.resources))
                return pod
        _LOGGER.warn(*self.log("failed to create a pod (%s)", request))
        return None

    def remove_pod(self, pod: Pod) -> None:
        self.nodes[pod.node_name].remove_pod(pod)

    def deploy_cluster(self, spec: ClusterSpec):
        self.cluster_quotas[spec.name] = deepcopy(spec.quota)
        cluster = Cluster(spec, self)
        self.clusters[spec.name] = cluster
        cluster.initialize()

    def get_quota(self, cluster_name: str) -> Resources:
        return self.cluster_quotas[cluster_name]

    def deploy_model(self, spec: InferenceModelSpec):
        cluster = self.clusters[spec.cluster_name]
        cluster.deploy_model(spec)

    def submit(self, task: Task):
        if isinstance(task, InferenceTask):
            self.clusters[task.cluster_name].submit(task)
        else:
            raise NotImplementedError()

    def run(self):
        # no action in the base controller
        try:
            yield self.env.timeout(float("inf"))
        except simpy.Interrupt:
            pass

    def stop(self):
        # graceful stop
        for cluster in self.clusters.values():
            yield self.env.process(cluster.stop())
        self.proc.interrupt()


class SimpleController(Controller):

    def run(self):
        while True:
            try:
                yield self.env.timeout(1)
            except simpy.Interrupt:
                break
            for cluster in self.clusters.values():
                for model in cluster.models.values():
                    avg_queue_len = np.mean(model.queue_len_metrics())
                    if avg_queue_len > 1:
                        _LOGGER.info(*self.log("add a replica from %s at %s",
                                               model.name, cluster.name))
                        model.add_replica()
                    if avg_queue_len == 0 and len(model.replicas) > 1:
                        _LOGGER.info(*self.log("remove a replica from %s at %s",
                                               model.name, cluster.name))
                        model.remove_replica()


# class SimAutoscaleCalculator(AutoscaleCalculator):

#     def __init__(self, head_name: str, min_replicas: int, max_replicas: int,
#                  worker_pod_resources: Resources, controller: Controller):
#         super().__init__(head_name, min_replicas, max_replicas, worker_pod_resources)
#         self.controller = controller

#     def try_allocation(self, num_requests: int, resource_request: Resources):
#         return self.controller.try_allocation(
#             cluster_name=self.head_name,
#             num_requests=num_requests,
#             resource_request=resource_request)


class AutoscaleExecutor:

    def __init__(self, cluster: Cluster, autoscale_config: AutoscaleConfig):
        self.env = global_env()
        self.cluster = cluster
        self.config = autoscale_config
        self.queue = simpy.Store(self.env)
        self.proc: simpy.Process = self.env.process(self.run())
        self.is_working = False
        # self.calculator = SimAutoscaleCalculator(
        #     controller=self.cluster.controller,
        #     head_name=self.cluster.name,
        #     min_replicas=self.config.min_replicas,
        #     max_replicas=self.config.max_replicas,
        #     worker_pod_resources=cluster.spec.worker_spec,
        # )

    def log(self, msg, *args):
        new_msg = "[%.3f] Autoscale executor (%s): " + msg
        new_args = (self.env.now, self.cluster.name, *args)
        return (new_msg, *new_args)

    def put(self, action: Action):
        self.queue.put(action)

    def run(self):
        while True:
            try:
                self.is_working = False
                request = self.queue.get()
                # action: Action = yield request
                action = yield request
                self.is_working = True

                dep = self.cluster.models[action.deployment_name]
                # new_num_replicas = self.calculator(action, dep)
                # Only target is supported for now by the executor.
                # The action should be proprocssed by check_action()
                new_num_replicas = action.target
                delta = new_num_replicas - dep.num_replicas
                if delta == 0:
                    _LOGGER.info(*self.log(
                        "No num_replica changes for [%s]%s: %d",
                        action.head_name, action.deployment_name,
                        new_num_replicas))
                else:
                    _LOGGER.info(*self.log(
                        "Changing num_replicas for [%s]%s from %d to %d",
                        action.head_name,
                        action.deployment_name,
                        dep.num_replicas,
                        new_num_replicas))
                    if delta > 0:
                        yield self.env.process(
                            dep.add_replica_generator(
                                delta,
                                # block until the action is completed.
                                max_trials=None,  # self.config.max_trials,
                                backoff_s=5.0  # self.config.backoff_s
                            )
                        )
                    else:
                        yield self.env.process(
                            dep.remove_replica_generator(
                                abs(delta),
                                self.config.downscale_action_delay_s))

                    _LOGGER.info(
                        *self.log("Changing replica for [%s]%s is done",
                                  action.head_name, action.deployment_name))
                    # apply drop rate if it exists
                    if action.accept_rate is not None:
                        dep.set_drop_rate(1 - action.accept_rate)

            except simpy.Interrupt:
                _LOGGER.debug(*self.log("interrupted"))
                request.cancel()
                break


class AutoscaleController(Controller):

    def __init__(self, policy_name: str, autoscale_config: Dict[str, Any],
                 input_rate_bin_size_s: float = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.autoscale_config = AutoscaleConfig.create(**autoscale_config)
        policy_params = kwargs.get("policy_params", {})
        _LOGGER.info("Autoscale policy: %s, params: %s, config: %s",
                     policy_name, policy_params, self.autoscale_config)
        self.policy: Policy = create_policy(
            policy_name, self.autoscale_config, **policy_params,
            input_rate_bin_size_s=input_rate_bin_size_s)
        self.executors: Dict[str, AutoscaleExecutor] = {}
        # for input rate metrics
        self.input_rate_bin_size_s = input_rate_bin_size_s
        self.arrival_ts_metric_stores = None

        self.state: State = None

    def generate_metrics(self):
        metrics = {}
        # average latency
        windows_start_ts = self.env.now - self.autoscale_config.look_back_period_s
        for cluster_name, cluster in self.clusters.items():
            # queue length & drop rate
            cluster_metrics = {}
            for model in cluster.models.values():
                cluster_metrics[model.name] = {
                    HEAD_QUEUE_LEN_KEY: model.get_queue_len(),
                    QUEUE_LEN_KEY: model.queue_len_metrics(),
                    DROP_RATE_KEY: model.drop_rate(),
                }
            # latency
            ts_tasks = metric.get_tasks(cluster_name, windows_start_ts)
            latency_dict = defaultdict(list)
            for ts_task in ts_tasks:
                if isinstance(ts_task.task, InferenceTask):
                    task = ts_task.task
                    latency_dict[task.model].append(task.elapsed_time() * 1e3)
                else:
                    raise NotImplementedError()
            for model_name, lats in latency_dict.items():
                cluster_metrics[model_name][LATENCY_KEY] = np.mean(lats)
            # input rate
            if self.arrival_ts_metric_stores is not None:
                hist_dict = self.arrival_ts_metric_stores[cluster_name].hist(
                    self.input_rate_bin_size_s, self.env.now)
                for model_name, hist in hist_dict.items():
                    cluster_metrics[model_name][INPUT_RATE_KEY] = hist
            head_name = cluster_name + "-ray-head"
            metrics[head_name] = cluster_metrics
        return metrics

    def submit(self, task: Task):
        if (self.arrival_ts_metric_stores is not None
                and isinstance(task, InferenceTask)):
            self.arrival_ts_metric_stores[task.cluster_name].record(
                task.model, self.env.now)
        return super().submit(task)

    def execute(self, action: Action):
        executor = self.executors[action.head_name]
        executor.put(action)

    @staticmethod
    def create_state(clusters: Dict[str, Cluster], nodes: Dict[str, Node]):
        num_replicas = defaultdict(dict)
        for cluster_name, cluster in clusters.items():
            for deployment_name, dep in cluster.models.items():
                head_name = f"{cluster_name}-ray-head"
                num_replicas[head_name][deployment_name] = dep.num_replicas
        # TODO: assume that resource limit in the policy is the same as the
        #       available resources in the cluster.

        available_quota = Resources()
        for node in nodes.values():
            available_quota += node.available_resources

        worker_pod_resource_dict = {
            f"{cluster_name}-ray-head": cluster.spec.worker_spec
            for cluster_name, cluster in clusters.items()
        }

        return State(
            num_replicas=num_replicas,
            available_quota=available_quota,
            worker_pod_resource_dict=worker_pod_resource_dict,
        )

    def execute_actions(self, actions: List[Action]):
        remained_actions = []
        actions = Autoscaler.sort_actions_by_delta(
            actions, self.state.num_replicas,
            self.state.autoscale_config.min_replicas,
            self.state.autoscale_config.max_replicas,
            only_downscale_first=True)

        for action in actions:
            if self.executors[action.head_name].is_working:
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
                executor = self.executors[action.head_name]
                action.trials = None
                executor.put(action)

        return remained_actions

    def run(self):
        for cluster_name, cluster in self.clusters.items():
            head_name = cluster_name + "-ray-head"
            self.executors[head_name] = AutoscaleExecutor(
                cluster, self.autoscale_config)
        if self.input_rate_bin_size_s is not None:
            _LOGGER.info("input rate metric is enabled. bin_size_s: %.2f",
                         self.input_rate_bin_size_s)
            self.arrival_ts_metric_stores = {
                cluster_name: TimeMetricStore()
                for cluster_name in self.clusters.keys()}

        remained_actions = []
        started = False
        self.state = self.create_state(self.clusters, self.nodes)
        self.state.autoscale_config = self.policy.autoscale_config
        _LOGGER.info("State: %s", self.state)
        custom_timeout = None
        while True:
            try:
                if not started and self.autoscale_config.interval_s > 60:
                    # long interval
                    # having short period at the beginning to have autoscale
                    yield self.env.timeout(60)
                    started = True
                    custom_timeout = self.autoscale_config.interval_s - 60
                else:
                    if custom_timeout is not None:
                        timeout = custom_timeout
                        custom_timeout = None
                    else:
                        timeout = self.autoscale_config.interval_s
                    yield self.env.timeout(timeout)
            except simpy.Interrupt:
                break
            metrics = self.generate_metrics()
            start_ts = time.time()
            actions = self.policy.autoscale(metrics, self.env.now)
            autoscale_time = time.time() - start_ts
            _LOGGER.info("Autoscale %s time: %.3f",
                         self.policy.__class__.__name__, autoscale_time)
            metric.autoscale_latency(autoscale_time)

            remained_actions.extend(actions)
            remained_actions = self.execute_actions(remained_actions)


class HybirdAutoscaleController(Controller):

    def __init__(self, policies: List[Dict[str, Any]],
                 input_rate_bin_size_s: float = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.executors: Dict[str, AutoscaleExecutor] = {}
        # for latency metrics
        self.look_back_period_s = None
        # for input rate metrics
        self.input_rate_bin_size_s = input_rate_bin_size_s
        self.arrival_ts_metric_stores = None

        # assume that policies are sorted by decending priority
        self.policies: List[Policy] = []
        interval_s_list = []
        for i, policy_config in enumerate(policies):
            policy_name = policy_config["policy_name"]
            autoscale_config = AutoscaleConfig.create(
                **policy_config["autoscale_config"])
            interval_s_list.append(autoscale_config.interval_s)
            if self.look_back_period_s is None:
                self.look_back_period_s = autoscale_config.look_back_period_s
            else:
                assert self.look_back_period_s == autoscale_config.look_back_period_s
            policy_params = policy_config.get("policy_params", {})
            _LOGGER.info(
                "Hybrid autoscale policy %d: %s, params: %s, config: %s",
                i, policy_name, policy_params, autoscale_config)
            self.policies.append(
                create_policy(policy_name, autoscale_config, **policy_params,
                              input_rate_bin_size_s=input_rate_bin_size_s)
            )

        # check intervals
        self.interval_s = min(interval_s_list)
        _LOGGER.info("interval_s: %d", self.interval_s)
        for interval_s in interval_s_list:
            if interval_s % self.interval_s != 0:
                raise ValueError(f"incompatible interval_s: {interval_s}")

        self.state: State = None

    def _get_metrics_for_not_working_clusters(self, metrics: Dict[str, dict]):
        # check whether autoscale actions are done
        return {
            head_name: cluster_metrics
            for head_name, cluster_metrics in metrics.items()
            if not self.executors[head_name].is_working
        }

    def generate_metrics(self, clear: bool):
        metrics = {}
        # average latency
        windows_start_ts = self.env.now - self.look_back_period_s
        for cluster_name, cluster in self.clusters.items():
            # queue length
            cluster_metrics = {}
            for model in cluster.models.values():
                cluster_metrics[model.name] = {
                    HEAD_QUEUE_LEN_KEY: model.get_queue_len(),
                    QUEUE_LEN_KEY: model.queue_len_metrics()}
            # latency
            ts_tasks = metric.get_tasks(cluster_name, windows_start_ts)
            latency_dict = defaultdict(list)
            for ts_task in ts_tasks:
                if isinstance(ts_task.task, InferenceTask):
                    task = ts_task.task
                    latency_dict[task.model].append(task.elapsed_time() * 1e3)
                else:
                    raise NotImplementedError()
            for model_name, lats in latency_dict.items():
                cluster_metrics[model_name][LATENCY_KEY] = np.mean(lats)
            # input rate
            if self.arrival_ts_metric_stores is not None:
                hist_dict = self.arrival_ts_metric_stores[cluster_name].hist(
                    self.input_rate_bin_size_s, self.env.now, clear)
                for model_name, hist in hist_dict.items():
                    cluster_metrics[model_name][INPUT_RATE_KEY] = hist
            head_name = cluster_name + "-ray-head"
            metrics[head_name] = cluster_metrics
        return metrics

    def submit(self, task: Task):
        if (self.arrival_ts_metric_stores is not None
                and isinstance(task, InferenceTask)):
            self.arrival_ts_metric_stores[task.cluster_name].record(
                task.model, self.env.now)
        return super().submit(task)

    def execute_actions(self, actions: List[Action]):
        remained_actions = []
        actions = Autoscaler.sort_actions_by_delta(
            actions, self.state.num_replicas,
            self.state.autoscale_config.min_replicas,
            self.state.autoscale_config.max_replicas,
            only_downscale_first=True)

        for action in actions:
            if self.executors[action.head_name].is_working:
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
                executor = self.executors[action.head_name]
                # update config for the current policy
                executor.config = self.state.autoscale_config
                # set trial to None to block until it is completed.
                action.trials = None
                executor.put(action)

        return remained_actions

    def run(self):
        for cluster_name, cluster in self.clusters.items():
            head_name = cluster_name + "-ray-head"
            self.executors[head_name] = AutoscaleExecutor(
                cluster, self.policies[0].autoscale_config)
        if self.input_rate_bin_size_s is not None:
            _LOGGER.info("input rate metric is enabled. bin_size_s: %.2f",
                         self.input_rate_bin_size_s)
            self.arrival_ts_metric_stores = {
                cluster_name: TimeMetricStore()
                for cluster_name in self.clusters.keys()}

        remained_actions = []
        started = False
        self.state = AutoscaleController.create_state(
            self.clusters, self.nodes)
        _LOGGER.info("State: %s", self.state)

        autoscale_time_s = 0
        while True:
            try:
                yield self.env.timeout(self.interval_s)
            except simpy.Interrupt:
                break

            for i, policy in enumerate(self.policies):
                run_autoscale = False
                if not started and policy.autoscale_config.interval_s > 60:
                    # long interval
                    # having short period at the beginning to have autoscale
                    if self.env.now % 60 == 0:
                        run_autoscale = True
                        started = True
                elif self.env.now % policy.autoscale_config.interval_s == 0:
                    run_autoscale = True

                if run_autoscale:
                    metrics = self.generate_metrics(i == 0)
                    if i > 0:
                        # assume that only 0-th policy is the utility policy
                        # disable autoscaling while the autoscaling action is in progress
                        metrics = self._get_metrics_for_not_working_clusters(
                            metrics)
                    start_ts = time.perf_counter()
                    actions = policy.autoscale(metrics, self.env.now)
                    if i == 0:
                        # highest level actions have more trials
                        for action in actions:
                            action.trials = 5
                    autoscale_time_s = ceil(time.perf_counter() - start_ts)
                    _LOGGER.info("Autoscale %s time: %.3f",
                                 policy.__class__.__name__, autoscale_time_s)
                    metric.autoscale_latency(autoscale_time_s)
                    break
            for policy in self.policies[i+1:]:
                # clear policies with lower priorities
                policy.clear()

            remained_actions.extend(actions)
            # update the state's config for the current policy
            self.state.autoscale_config = policy.autoscale_config
            remained_actions = self.execute_actions(remained_actions)


_CLASS_MAPPING = {
    "default": Controller,
    "simple": SimpleController,
    "autoscale": AutoscaleController,
    "hybrid": HybirdAutoscaleController,
}


def create_controller(type: str, *args, **kwargs) -> Controller:
    return _CLASS_MAPPING[type](*args, **kwargs)
