from abc import abstractmethod
from collections import defaultdict
import functools
import json
import logging
from math import ceil
from pathlib import Path
from typing import Callable, DefaultDict, List, Dict, Optional, Tuple

import numpy as np

from predictor import OraclePredictor, Predictor
import autoscale
from autoscale import Action, AutoscaleConfig, create_metric_parser
from solver import Solver, SolverWithDrop, HierarchicalSolver, HierarchicalSolverWithDrop
from utility import create_utility_func

_LOGGER = logging.getLogger(__name__)


class Policy:

    @abstractmethod
    def autoscale(self, input_metrics, current_ts: float = None) -> List[Action]:
        pass

    @abstractmethod
    def clear(self):
        pass


class BasePolicy(Policy):

    def __init__(self, autoscale_config: AutoscaleConfig,
                 input_rate_bin_size_s: int):
        # TODO: support per-cluster config
        self.autoscale_config = autoscale_config
        self.counts = defaultdict(lambda: defaultdict(int))
        self.metric_parser = create_metric_parser(autoscale_config.metric)

    def clear(self):
        return self.counts.clear()

    def get_target_metric(self, head_name: str) -> float:
        if self.autoscale_config.cluster_configs is None:
            return self.autoscale_config.target_metric
        else:
            return self.autoscale_config.cluster_configs[head_name].target_metric

    @abstractmethod
    def generate_upscale_action(self, metric: float, head_name: str,
                                deployment_name: str) -> Action:
        pass

    @abstractmethod
    def generate_downscale_action(self, metric: float, head_name: str,
                                  deployment_name: str) -> Action:
        pass

    def upscale(self, head_name: str, deployment_name: str,
                current_metric: float) -> List[Action]:
        actions = []
        count = self.counts[head_name][deployment_name]
        count = 1 if count < 0 else count + 1
        threshold = self.autoscale_config.upscale_threshold
        _LOGGER.debug(
            "Increment upscale count for [%s]%s : %d/%d, "
            "current=%.2f",
            head_name, deployment_name, count, threshold,
            current_metric)
        if count >= threshold:
            actions.append(self.generate_upscale_action(
                metric=current_metric,
                head_name=head_name,
                deployment_name=deployment_name))
            count = 0  # reset count
        self.counts[head_name][deployment_name] = count
        return actions

    def downscale(self, head_name: str, deployment_name: str,
                  current_metric: float) -> List[Action]:
        actions = []
        count = self.counts[head_name][deployment_name]
        count = -1 if count > 0 else count - 1
        threshold = self.autoscale_config.downscale_threshold
        _LOGGER.debug(
            "Increment downscale count for [%s]%s : %d/%d, "
            "current=%.2f",
            head_name, deployment_name, -count, threshold,
            current_metric)
        if -count >= threshold:
            actions.append(self.generate_downscale_action(
                metric=current_metric,
                head_name=head_name,
                deployment_name=deployment_name))
            count = 0  # reset count
        self.counts[head_name][deployment_name] = count
        return actions

    def autoscale(self, input_metrics, current_ts: float = None) -> List[Action]:
        actions = []
        for head_name, cluster_metrics in input_metrics.items():
            for deployment_name, metrics in cluster_metrics.items():
                current_metric = self.metric_parser(metrics)
                if current_metric > self.get_target_metric(head_name):
                    # upscale
                    actions.extend(self.upscale(head_name, deployment_name,
                                                current_metric))
                else:
                    # downscale
                    actions.extend(self.downscale(head_name, deployment_name,
                                                  current_metric))
        return actions


class NonePolicy(BasePolicy):
    """Do-nothing policy.

    This is for monitoring.
    """

    def generate_downscale_action(self, metric: float, head_name: str, deployment_name: str) -> Action:
        raise NotImplementedError()

    def generate_upscale_action(self, metric: float, head_name: str, deployment_name: str) -> Action:
        return NotImplementedError()

    def autoscale(self, input_metrics, current_ts: float = None) -> List[Action]:
        return []


class OneshotPolicy(BasePolicy):
    """Oneshot policy

    This policy up/downscales replicas based on a ratio of the current metrics
    over the target metrics.
    """

    def generate_upscale_action(self, metric: float, head_name: str,
                                deployment_name: str) -> Action:
        factor = metric / self.get_target_metric(head_name)
        _LOGGER.info("Upscale for [%s]%s. factor=%f",
                     head_name, deployment_name, factor)
        return Action(head_name=head_name,
                      deployment_name=deployment_name,
                      factor=factor)

    @abstractmethod
    def generate_downscale_action(self, metric: float, head_name: str,
                                  deployment_name: str) -> Action:
        factor = metric / self.get_target_metric(head_name)
        _LOGGER.info("Downscale for [%s]%s. factor=%f",
                     head_name, deployment_name, factor)
        return Action(head_name=head_name,
                      deployment_name=deployment_name,
                      factor=factor)


class OneshotEMAPolicy(BasePolicy):
    """Oneshot exponential moving average policy

    This policy up/downscales replicas based on a ratio of the EMA metrics
    over the target metrics.

    Upscale and downscale demand is based on the current metric.
    Using EMA to avoid over-reaction from the delay.
    """

    def __init__(self, autoscale_config: AutoscaleConfig,
                 input_rate_bin_size_s: int):
        super().__init__(autoscale_config, input_rate_bin_size_s)
        self.alpha = 0.3
        self.emas = defaultdict(dict)
        _LOGGER.info("EMA alpha: %s", self.alpha)

    def update_ema(self, metric: float, head_name: str, deployment_name: str):
        cluster_ema = self.emas[head_name]
        if deployment_name not in cluster_ema:
            cluster_ema[deployment_name] = metric
        else:
            cluster_ema[deployment_name] = self.alpha * \
                cluster_ema[deployment_name] + (1 - self.alpha) * metric
        _LOGGER.info(
            "EMA for [%s]%s: %.2f",
            head_name, deployment_name, cluster_ema[deployment_name])

    def generate_upscale_action(self, metric: float, head_name: str,
                                deployment_name: str) -> Action:
        factor = self.emas[head_name][deployment_name] / \
            self.get_target_metric(head_name)
        _LOGGER.info("Upscale for [%s]%s. factor=%f",
                     head_name, deployment_name, factor)
        return Action(head_name=head_name,
                      deployment_name=deployment_name,
                      factor=factor)

    @abstractmethod
    def generate_downscale_action(self, metric: float, head_name: str,
                                  deployment_name: str) -> Action:
        factor = self.emas[head_name][deployment_name] / \
            self.get_target_metric(head_name)
        _LOGGER.info("Downscale for [%s]%s. factor=%f",
                     head_name, deployment_name, factor)
        return Action(head_name=head_name,
                      deployment_name=deployment_name,
                      factor=factor)

    def autoscale(self, input_metrics, current_ts: float = None) -> List[Action]:
        actions = []
        for head_name, cluster_metrics in input_metrics.items():
            for deployment_name, metrics in cluster_metrics.items():
                current_metric = self.metric_parser(metrics)
                self.update_ema(current_metric, head_name, deployment_name)
                if current_metric > self.get_target_metric(head_name):
                    # upscale
                    actions.extend(self.upscale(head_name, deployment_name,
                                                current_metric))
                else:
                    # downscale
                    actions.extend(self.downscale(head_name, deployment_name,
                                                  current_metric))
        return actions


class AIADPolicy(BasePolicy):
    """Addictive increase and addictive decrease policy."""

    def generate_upscale_action(self, metric: float, head_name: str,
                                deployment_name: str) -> Action:
        _LOGGER.info("Upscale for [%s]%s. delta=%d, metric=%.2f",
                     head_name, deployment_name, 1, metric)
        return Action(head_name=head_name,
                      deployment_name=deployment_name,
                      delta=1)

    def generate_downscale_action(self, metric: float, head_name: str,
                                  deployment_name: str) -> Action:
        _LOGGER.info("Downscale for [%s]%s. delta=%d, metric=%.2f",
                     head_name, deployment_name, -1, metric)
        return Action(head_name=head_name,
                      deployment_name=deployment_name,
                      delta=-1)


class AIPolicy(AIADPolicy):
    """Addictive increase policy."""

    def downscale(self, head_name: str, deployment_name: str,
                  current_metric: float) -> List[Action]:
        # no downscale
        return []


class OMIOMDPolicy(BasePolicy):
    """One-shot multiplicative increase and one-shot multiplicative decrease

    Calculate delta based on a ratio of current metric over target metric,
    but each step is capped by a factor that increases in a multiplicative
    manner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upscale_limit: Dict[str, int] = {}
        self.downscale_limit: Dict[str, int] = {}

    @staticmethod
    def _get_key(head_name: str, deployment_name: str) -> str:
        return f"{head_name}#{deployment_name}"

    def get_upscale_limit(self, head_name: str, deployment_name: str) -> int:
        key = self._get_key(head_name, deployment_name)
        # reset downscale limit
        self.downscale_limit[key] = 1
        limit = self.upscale_limit.get(key, 1)
        self.upscale_limit[key] = limit * 2
        return limit

    def get_downscale_limit(self, head_name: str, deployment_name: str) -> int:
        key = self._get_key(head_name, deployment_name)
        # reset upscale limit
        self.upscale_limit[key] = 1
        limit = self.downscale_limit.get(key, 1)
        self.downscale_limit[key] = limit * 2
        return limit

    def generate_upscale_action(self, metric: float, head_name: str,
                                deployment_name: str) -> Action:
        factor = metric / self.get_target_metric(head_name)
        limit = self.get_upscale_limit(head_name, deployment_name)
        _LOGGER.info("Upscale for [%s]%s. factor=%f, limit=%d",
                     head_name, deployment_name, factor, limit)
        return Action(head_name=head_name,
                      deployment_name=deployment_name,
                      factor=factor,
                      limit=limit)

    def generate_downscale_action(self, metric: float, head_name: str,
                                  deployment_name: str) -> Action:
        factor = metric / self.get_target_metric(head_name)
        limit = self.get_downscale_limit(head_name, deployment_name)
        _LOGGER.info("Downscale for [%s]%s. factor=%f, limit=%d",
                     head_name, deployment_name, factor, limit)
        return Action(head_name=head_name,
                      deployment_name=deployment_name,
                      factor=factor,
                      limit=limit)


class BasePredictionPolicy(Policy):

    _AGGREGATOR = {
        "avg": np.average,
        "max": np.max,
        "p80": functools.partial(np.percentile, q=80),
        "none": lambda x: np.array(x),  # do nothing
    }

    @staticmethod
    def calculate_latency_estimate(estimator_type, aggregated_input_rate):
        if estimator_type == "max":
            return aggregated_input_rate
        elif estimator_type == "avg":
            return aggregated_input_rate / 2
        elif estimator_type.startswith("p"):
            factor = int(estimator_type[1:]) / 100
            return factor * aggregated_input_rate
        else:
            raise ValueError("Unknown estimator type: %s" % estimator_type)

    def __init__(self, autoscale_config: AutoscaleConfig, window_size_s: int,
                 input_rate_bin_size_s: int, aggregator="avg",
                 latency_estimator: str = "max",
                 processing_time: float = 180,
                 cluster_processing_times: Dict[str, float] = None,
                 use_current_queue_len: bool = False,
                 queue_len_limit: int = 400):
        self.autoscale_config = autoscale_config
        self.aggregator = self._AGGREGATOR[aggregator]
        self.input_rate_parser = create_metric_parser("input_rate")
        self.window_size_s = window_size_s
        self.input_rate_bin_size_s = input_rate_bin_size_s
        self.predictors = self.build_predictors()
        self.latency_estimator = latency_estimator
        self.use_current_queue_len = use_current_queue_len
        self.processing_time = processing_time
        self.cluster_processing_times = cluster_processing_times
        self.queue_len_limit = queue_len_limit
        self.queue_len_parser = create_metric_parser("total_queue_len")

        _LOGGER.info("use_current_queue_len: %s", use_current_queue_len)

    def get_processing_time(self, head_name: str) -> float:
        if self.cluster_processing_times is None:
            return self.processing_time
        else:
            return self.cluster_processing_times[head_name]

    def get_target_metric(self, head_name: str) -> float:
        if self.autoscale_config.cluster_configs is None:
            return self.autoscale_config.target_metric
        else:
            return self.autoscale_config.cluster_configs[head_name].target_metric

    @abstractmethod
    def build_predictors(self) -> DefaultDict[str, Dict[str, Predictor]]:
        pass

    def calculate_estimate(self, next_num_inputs,
                           current_num_inputs, current_ts,
                           head_name, deployment_name, total_queue_len):
        _LOGGER.info("[%.3f] (%s:%s) current_queue_len: %d", current_ts,
                     head_name, deployment_name, total_queue_len)
        if next_num_inputs is None:
            # use the current_num_inputs if it exists.
            if current_num_inputs is None:
                next_num_inputs = [0]
            else:
                next_num_inputs, _ = current_num_inputs
            _LOGGER.info(
                "[%.3f] (%s:%s) Not available next num inputs. "
                "Use current input rate. len=%d",
                current_ts, head_name, deployment_name, len(next_num_inputs))

        if self.use_current_queue_len:
            # TODO: can we do better when self.input_rate_bin_size_s > 1
            # assert self.input_rate_bin_size_s == 1
            if (self.queue_len_limit > 0 and
                    total_queue_len > self.queue_len_limit and
                    self.input_rate_bin_size_s == 1):
                _LOGGER.info(
                    "Too large queue length (%d) prevents appropriate "
                    "optimization. limit to %s",
                    total_queue_len, self.queue_len_limit)
                total_queue_len = self.queue_len_limit
            next_num_inputs = np.concatenate(
                ([total_queue_len], next_num_inputs))

        aggregated = self.aggregator(next_num_inputs)
        estimate = self.calculate_latency_estimate(
            self.latency_estimator, aggregated)
        _LOGGER.info(
            "(%s:%s) aggregated=%s, len=%d, estimate=%s",
            head_name, deployment_name,
            aggregated, len(next_num_inputs), estimate)

        return estimate / self.input_rate_bin_size_s

    def autoscale(self, input_metrics, current_ts: float = None) -> List[Action]:
        actions = []
        for head_name, cluster_metrics in input_metrics.items():
            for deployment_name, metrics in cluster_metrics.items():
                _LOGGER.info("[%.3f] (%s:%s) metrics: %s",
                             current_ts, head_name, deployment_name, metrics)
                predictor = self.predictors[head_name][deployment_name]
                current_num_inputs = self.input_rate_parser(metrics)
                next_num_inputs = predictor.predict(
                    current_num_inputs, current_ts)

                estimate = self.calculate_estimate(
                    next_num_inputs, current_num_inputs, current_ts,
                    head_name, deployment_name, self.queue_len_parser(metrics))

                required_num_replicas = ceil(
                    estimate * self.get_processing_time(head_name) / self.get_target_metric(head_name))
                _LOGGER.info(
                    "(%s:%s) num_replicas_demand: %s",
                    head_name, deployment_name, required_num_replicas)

                actions.append(Action(
                    head_name=head_name,
                    deployment_name=deployment_name,
                    target=required_num_replicas))
        return actions


class OraclePredictionPolicy(BasePredictionPolicy):

    def __init__(self, autoscale_config: AutoscaleConfig, input_path: str,
                 window_size_s: int, input_rate_bin_size_s: int,
                 aggregator: str = "avg", latency_estimator: str = "max",
                 processing_time: float = 180,
                 use_current_queue_len: bool = False,
                 queue_len_limit: int = 400):
        self.input_path = input_path
        super().__init__(
            autoscale_config=autoscale_config, window_size_s=window_size_s,
            input_rate_bin_size_s=input_rate_bin_size_s, aggregator=aggregator,
            latency_estimator=latency_estimator, processing_time=processing_time,
            use_current_queue_len=use_current_queue_len,
            queue_len_limit=queue_len_limit)

    def build_predictors(self) -> DefaultDict[str, Dict[str, OraclePredictor]]:
        lines = Path(self.input_path).read_text().splitlines()
        counts_dict = defaultdict(lambda: defaultdict(list))
        for line in lines:
            obj = json.loads(line)
            for cluster_name, jobs in obj["counts"].items():
                for job_name, count in jobs.items():
                    counts_dict[cluster_name][job_name].append(count)
        predictors = defaultdict(dict)
        for cluster_name, jobs_counts in counts_dict.items():
            for job_name, counts in jobs_counts.items():
                head_name = cluster_name + "-ray-head"
                predictors[head_name][job_name] = OraclePredictor(
                    input_counts=counts,
                    window_size_s=self.window_size_s,
                    head_name=head_name,
                    deployment_name=job_name,
                    unit_time_s=self.input_rate_bin_size_s,
                )
        return predictors


class PyTorchForecastingPredictionPolicy(BasePredictionPolicy):

    def __init__(self, autoscale_config: AutoscaleConfig, window_size_s: int,
                 input_rate_bin_size_s: int,
                 model_params: Dict[str, Dict[str, Dict]],
                 aggregator: str = "avg", latency_estimator: str = "max",
                 processing_time: float = 180,
                 use_current_queue_len: bool = False,
                 queue_len_limit: int = 400):
        self.model_params = model_params
        super().__init__(
            autoscale_config=autoscale_config, window_size_s=window_size_s,
            input_rate_bin_size_s=input_rate_bin_size_s, aggregator=aggregator,
            latency_estimator=latency_estimator, processing_time=processing_time,
            use_current_queue_len=use_current_queue_len,
            queue_len_limit=queue_len_limit)

    def build_predictors(self) -> DefaultDict[str, Dict[str, Predictor]]:
        from ptf_predictor import PyTorchForcastingPredictor
        predictors = defaultdict(dict)
        for cluster_name, jobs in self.model_params.items():
            for job_name, model_param in jobs.items():
                head_name = cluster_name + "-ray-head"
                predictors[head_name][job_name] = \
                    PyTorchForcastingPredictor(
                        window_size_s=self.window_size_s,
                        head_name=head_name,
                        deployment_name=job_name,
                        **model_param)
        return predictors


class DartsPredictionPolicy(BasePredictionPolicy):

    def __init__(self, autoscale_config: AutoscaleConfig, window_size_s: int,
                 input_rate_bin_size_s: int,
                 model_params: Dict[str, Dict[str, Dict]],
                 aggregator: str = "avg", latency_estimator: str = "max",
                 processing_time: float = 180,
                 cluster_processing_times: Dict[str, float] = None,
                 use_current_queue_len: bool = False,
                 queue_len_limit: int = 400):
        self.model_params = model_params
        super().__init__(
            autoscale_config=autoscale_config, window_size_s=window_size_s,
            input_rate_bin_size_s=input_rate_bin_size_s, aggregator=aggregator,
            latency_estimator=latency_estimator, processing_time=processing_time,
            cluster_processing_times=cluster_processing_times,
            use_current_queue_len=use_current_queue_len,
            queue_len_limit=queue_len_limit)

    def build_predictors(self) -> DefaultDict[str, Dict[str, Predictor]]:
        from darts_predictor import DartsPredictor
        predictors = defaultdict(dict)
        for cluster_name, jobs in self.model_params.items():
            for job_name, model_param in jobs.items():
                head_name = cluster_name + "-ray-head"
                predictors[head_name][job_name] = \
                    DartsPredictor(window_size_s=self.window_size_s,
                                   head_name=head_name,
                                   deployment_name=job_name,
                                   unit_time_s=self.input_rate_bin_size_s,
                                   **model_param)
        return predictors


class UtilityPolicy(Policy):

    def __init__(self, autoscale_config: AutoscaleConfig, util_type: str,
                 resource_limit: dict, input_rate_bin_size_s: int,
                 adjust: bool = False, min_max: bool = False,
                 method: str = "slsqp", individually: bool = False,
                 processing_time: float = 180,
                 cluster_processing_times: Dict[str, float] = None,
                 weighted_util: bool = False,
                 weighted_util_scale: bool = False,
                 with_drop: bool = False,
                 changes_weight: float = 0.0,
                 drop_integrality: bool = False,
                 drop_weight: float = 0.0,
                 linear_penalty: bool = False,
                 mdc_percentile: int = 0,
                 utility_weight: Optional[float] = None):
        self.autoscale_config = autoscale_config
        self.util_type = util_type
        self.utility_func = create_utility_func(
            util_type, autoscale_config.target_metric)
        if self.autoscale_config.cluster_configs is not None:
            self.cluster_utility_func = {
                head_name: create_utility_func(
                    util_type, cluster_config.target_metric)
                for head_name, cluster_config in
                self.autoscale_config.cluster_configs.items()}
        else:
            self.cluster_utility_func = None
        self.metric_parser = create_metric_parser(autoscale_config.metric)
        self.input_rate_parser = create_metric_parser("input_rate")
        self.resource_limit = resource_limit
        self.min_max = min_max
        self.method = method
        self.adjust = adjust
        self.individually = individually
        self.processing_time = processing_time
        self.cluster_processing_times = cluster_processing_times
        self.weighted_util = weighted_util
        self.weighted_util_scale = weighted_util_scale
        self.with_drop = with_drop
        self.solver_cls = SolverWithDrop if with_drop else Solver
        self.changes_weight = changes_weight
        self.drop_integrality = drop_integrality
        self.drop_weight = drop_weight
        self.linear_penalty = linear_penalty
        self.mdc_percentile = mdc_percentile
        self.utility_weight = utility_weight

        if self.individually:
            assert not self.min_max

        _LOGGER.info(
            "Utility policy: type=%s, resource_limit=%s, adjust=%s, min_max=%s"
            ", individually=%s, processing_time=%.2f, weighted_util=%s"
            ", weighted_util_scale=%s, with_drop=%s, changes_weight=%f"
            ", drop_integrality=%s, drop_weight=%.2g, linear_penalty=%s"
            ", mdc_percentile=%s, utility_weight=%s, cluster_processing_times=%s",
            util_type, resource_limit, adjust, min_max, individually,
            processing_time, weighted_util, weighted_util_scale, with_drop,
            changes_weight, drop_integrality, drop_weight, linear_penalty,
            mdc_percentile, utility_weight, cluster_processing_times)

    def get_utility_func(self, head_name: str) -> Callable[[float], float]:
        if self.cluster_utility_func is not None:
            return self.cluster_utility_func[head_name]
        else:
            return self.utility_func

    def get_processing_time(self, head_name: str) -> float:
        if self.cluster_processing_times is None:
            return self.processing_time
        else:
            return self.cluster_processing_times[head_name]

    def create_solver(self):
        return self.solver_cls(resource_limit=self.resource_limit["cpu"],
                               adjust=self.adjust,
                               min_max=self.min_max,
                               method=self.method,
                               changes_weight=self.changes_weight,
                               drop_integrality=self.drop_integrality,
                               drop_weight=self.drop_weight,
                               linear_penalty=self.linear_penalty,
                               mdc_percentile=self.mdc_percentile,
                               utility_weight=self.utility_weight,
                               util_type=self.util_type)

    def autoscale(self, input_metrics, current_ts: float = None) -> List[Action]:
        if len(input_metrics) == 0:
            return []

        utilities = {}
        solver_kwargs_list = []

        for head_name, cluster_metrics in input_metrics.items():
            for deployment_name, metrics in cluster_metrics.items():
                utility_fn = self.get_utility_func(head_name)
                utility = utility_fn(self.metric_parser(metrics))
                utilities[(head_name, deployment_name)] = utility

                input_rate = self.input_rate_parser(metrics)
                if input_rate is None:
                    input_rate = 0.0
                else:
                    counts, _ = input_rate
                    input_rate = np.sum(counts) / \
                        self.autoscale_config.interval_s
                _LOGGER.info("input rate for [%s]%s: %.2f",
                             head_name, deployment_name, input_rate)
                num_replicas = len(metrics[autoscale.QUEUE_LEN_KEY])
                solver_kwargs_list.append(kwargs_to_dict(
                    key=(head_name, deployment_name),
                    processing_time=self.get_processing_time(head_name),
                    input_rate=input_rate,
                    weight=1.0,
                    slo_target=utility_fn.target_latency,
                    current_num_replicas=num_replicas,
                    resource_per_replica=1,  # only consider cpu for now
                ))
                if self.with_drop:
                    solver_kwargs_list[-1]["accept_rate"] = 1 - metrics.get(
                        autoscale.DROP_RATE_KEY, 0.0)

        if self.individually:
            next_num_replicas = {}
            new_utility = []
            for solver_kwargs in solver_kwargs_list:
                solver = self.create_solver()
                solver.add_deployment(**solver_kwargs)
                next_num_replicas_item, new_utility_item = solver.solve()
                next_num_replicas.update(next_num_replicas_item)
                new_utility.append(new_utility_item)

        else:
            solver = self.create_solver()
            if self.weighted_util:
                input_rates = np.array([
                    solver_kwargs["input_rate"]
                    for solver_kwargs in solver_kwargs_list])
                weights = input_rates / np.sum(input_rates)
                if self.weighted_util_scale:
                    weights *= len(input_rates)

                for weight, solver_kwargs in zip(weights, solver_kwargs_list):
                    solver_kwargs["weight"] = weight

            for solver_kwargs in solver_kwargs_list:
                solver.add_deployment(**solver_kwargs)

            next_num_replicas, new_utility = solver.solve()

        _LOGGER.info("[%.3f] current utilities: %s, sum: %.2f",
                     current_ts, str(utilities), sum(utilities.values()))
        _LOGGER.info("[%.3f] new utility sum: %.2f, new_num_replicas: %s",
                     current_ts, np.sum(new_utility), next_num_replicas)
        actions = []
        for (head_name, deployment_name), num_replicas in next_num_replicas.items():
            actions.append(Action(head_name=head_name,
                                  deployment_name=deployment_name,
                                  target=num_replicas))
        return actions


def kwargs_to_dict(**kwargs):
    return kwargs


class PredictionUtilityAutoscaler:

    def __init__(self, predictors, calculate_estimate,
                 queue_len_parser, window_size_s, input_rate_bin_size_s,
                 autoscale_config: AutoscaleConfig,
                 util_type: str,
                 resource_limit: dict,
                 input_path: str = None,
                 adjust: bool = False,
                 individually: bool = False,
                 min_max: bool = False,
                 method: str = "slsqp",
                 processing_time: float = 180,
                 cluster_processing_times: Dict[str, float] = None,
                 weighted_util: bool = False,
                 weighted_util_scale: bool = False,
                 skip_without_pred: bool = False,
                 with_drop: bool = False,
                 changes_weight: float = 0.0,
                 drop_integrality: bool = False,
                 drop_weight: float = 0.0,
                 linear_penalty: bool = False,
                 mdc_percentile: int = 0,
                 utility_weight: Optional[float] = None,
                 upscale_overhead: int = 0,
                 max_rho=0.95,
                 slack_resources: Optional[dict] = None,
                 hierarchical: Optional[int] = None):
        self.predictors = predictors
        self.calculate_estimate = calculate_estimate
        self.queue_len_parser = queue_len_parser
        self.window_size_s = window_size_s
        self.input_rate_bin_size_s = input_rate_bin_size_s
        self.input_path = input_path

        self.metric_parser = create_metric_parser(autoscale_config.metric)
        self.input_rate_parser = create_metric_parser("input_rate")
        self.util_type = util_type
        self.utility_func = create_utility_func(
            util_type, autoscale_config.target_metric)
        if autoscale_config.cluster_configs is not None:
            self.cluster_utility_func = {
                head_name: create_utility_func(
                    util_type, cluster_config.target_metric)
                for head_name, cluster_config in
                autoscale_config.cluster_configs.items()}
        else:
            self.cluster_utility_func = None
        self.resource_limit = resource_limit
        self.adjust = adjust
        self.min_max = min_max
        self.method = method
        self.individually = individually
        self.processing_time = processing_time
        self.cluster_processing_times = cluster_processing_times
        self.weighted_util = weighted_util
        self.weighted_util_scale = weighted_util_scale
        self.skip_without_pred = skip_without_pred
        self.with_drop = with_drop
        if hierarchical is not None:
            self.solver_cls = HierarchicalSolverWithDrop if with_drop else HierarchicalSolver
        else:
            self.solver_cls = SolverWithDrop if with_drop else Solver
        self.changes_weight = changes_weight
        self.drop_integrality = drop_integrality
        self.drop_weight = drop_weight
        self.linear_penalty = linear_penalty
        self.mdc_percentile = mdc_percentile
        self.utility_weight = utility_weight
        self.upscale_overhead = upscale_overhead
        self.max_rho = max_rho
        self.slack_resources = slack_resources
        self.hierarchical = hierarchical

        self.oracle_predictors = defaultdict(dict)
        if input_path is not None:
            lines = Path(input_path).read_text().splitlines()
            counts_dict = defaultdict(lambda: defaultdict(list))
            for line in lines:
                obj = json.loads(line)
                for cluster_name, jobs in obj["counts"].items():
                    for job_name, count in jobs.items():
                        counts_dict[cluster_name][job_name].append(count)
            for cluster_name, jobs_counts in counts_dict.items():
                head_name = cluster_name + "-ray-head"
                for job_name, counts in jobs_counts.items():
                    self.oracle_predictors[head_name][job_name] = OraclePredictor(
                        input_counts=counts,
                        window_size_s=self.window_size_s,
                        head_name=cluster_name,
                        deployment_name=job_name,
                        unit_time_s=self.input_rate_bin_size_s,
                    )
            self.latency_parser = create_metric_parser("avg_latency")

    def config_str(self):
        return ", ".join([
            f"type={self.util_type}",
            f"resource_limit={self.resource_limit}",
            f"adjust={self.adjust}",
            f"min_max={self.min_max}",
            f"processing_time={self.processing_time:.2f}",
            f"cluster_processing_times={self.cluster_processing_times}",
            f"weighted_util={self.weighted_util}",
            f"weighted_util_scale={self.weighted_util_scale}",
            f"with_drop={self.with_drop}",
            f"changes_weight={self.changes_weight}",
            f"input_path={self.input_path}",
            f"drop_integrality={self.drop_integrality}",
            f"drop_weight={self.drop_weight}",
            f"linear_penalty={self.linear_penalty}",
            f"mdc_percentile={self.mdc_percentile}",
            f"utility_weight={self.utility_weight}",
            f"upscale_overhead={self.upscale_overhead}",
            f"max_rho={self.max_rho}",
            f"slack_resources={self.slack_resources}",
            f"hierarchical={self.hierarchical}",
        ])

    def get_utility_func(self, head_name: str) -> Callable[[float], float]:
        if self.cluster_utility_func is not None:
            return self.cluster_utility_func[head_name]
        else:
            return self.utility_func

    def get_processing_time(self, head_name):
        if self.cluster_processing_times is not None:
            return self.cluster_processing_times[head_name]
        else:
            return self.processing_time

    def create_solver(self):
        return self.solver_cls(resource_limit=self.resource_limit["cpu"],
                               adjust=self.adjust,
                               min_max=self.min_max,
                               method=self.method,
                               changes_weight=self.changes_weight,
                               drop_integrality=self.drop_integrality,
                               drop_weight=self.drop_weight,
                               linear_penalty=self.linear_penalty,
                               mdc_percentile=self.mdc_percentile,
                               utility_weight=self.utility_weight,
                               upscale_overhead=int(
                                   self.upscale_overhead/self.input_rate_bin_size_s),
                               max_rho=self.max_rho,
                               num_children=self.hierarchical,
                               util_type=self.util_type)

    def calculate_utilities(self, input_metrics) -> Dict[Tuple[str, str], float]:
        utilities = {}
        for head_name, cluster_metrics in input_metrics.items():
            for deployment_name, metrics in cluster_metrics.items():
                utility_fn = self.get_utility_func(head_name)
                utility = utility_fn(self.metric_parser(metrics))
                utilities[(head_name, deployment_name)] = utility
        return utilities

    def solve_optim(self, input_metrics, current_ts: float = None):
        solver_kwargs_list = []
        next_num_input_list = []

        for head_name, cluster_metrics in input_metrics.items():
            for deployment_name, metrics in cluster_metrics.items():
                predictor = self.predictors[head_name][deployment_name]
                current_num_inputs = self.input_rate_parser(metrics)
                next_num_inputs = predictor.predict(
                    current_num_inputs, current_ts)
                next_num_input_list.append(next_num_inputs)

                if next_num_inputs is None and self.skip_without_pred:
                    _LOGGER.info(
                        "[%.3f] (%s:%s) Not available next num inputs.",
                        current_ts, head_name, deployment_name)
                    continue

                estimate = self.calculate_estimate(
                    next_num_inputs, current_num_inputs, current_ts,
                    head_name, deployment_name, self.queue_len_parser(metrics))

                oracle_predictor: OraclePredictor = \
                    self.oracle_predictors[head_name].get(
                        deployment_name, None)
                if oracle_predictor is not None:
                    oracle_next_num_inputs = oracle_predictor.predict(
                        current_num_inputs, current_ts)
                    oracle_estimate = self.calculate_estimate(
                        oracle_next_num_inputs, current_num_inputs, current_ts,
                        head_name, deployment_name, self.queue_len_parser(metrics))
                    if isinstance(estimate, float):
                        error = estimate - oracle_estimate
                    else:  # array
                        # either estimate or oracle estimate may have
                        # shorter length
                        # - shorter estimate: when using history due to
                        #   insufficient history length
                        # - shorter oracle: when the input traces end.
                        # use shorter length among them
                        target_len = min([len(estimate), len(oracle_estimate)])
                        error = estimate[:target_len] - \
                            oracle_estimate[:target_len]
                    _LOGGER.info(
                        "[%.3f] (%s:%s) pred_error: %s, estimate: %s (oracle: %s), avg_lat: %.2f",
                        current_ts, head_name, deployment_name,
                        error, estimate, oracle_estimate,
                        self.latency_parser(metrics))

                num_replicas = len(metrics[autoscale.QUEUE_LEN_KEY])
                solver_kwargs_list.append(kwargs_to_dict(
                    key=(head_name, deployment_name),
                    processing_time=self.get_processing_time(head_name),
                    input_rate=estimate,
                    weight=1.0,
                    slo_target=self.get_utility_func(head_name).target_latency,
                    current_num_replicas=num_replicas,
                    resource_per_replica=1,  # only consider cpu for now
                ))
                if self.with_drop:
                    solver_kwargs_list[-1]["accept_rate"] = 1 - metrics.get(
                        autoscale.DROP_RATE_KEY, 0.0)

        if len(solver_kwargs_list) == 0:
            return None

        if self.individually:
            next_num_replicas = {}
            new_utility = []
            for solver_kwargs in solver_kwargs_list:
                solver = self.create_solver()
                solver.add_deployment(**solver_kwargs)
                next_num_replicas_item, new_utility_item = solver.solve()
                next_num_replicas.update(next_num_replicas_item)
                new_utility.append(new_utility_item)
        else:
            solver = self.create_solver()
            if self.weighted_util:
                input_rates = np.array([
                    solver_kwargs["input_rate"]
                    for solver_kwargs in solver_kwargs_list])
                input_rate_sum = np.sum(input_rates)
                if input_rate_sum > 0:
                    weights = input_rates / input_rate_sum
                else:
                    weights = np.ones_like(input_rates) / len(input_rates)
                if self.weighted_util_scale:
                    weights *= len(input_rates)

                for weight, solver_kwargs in zip(weights, solver_kwargs_list):
                    solver_kwargs["weight"] = weight

            for solver_kwargs in solver_kwargs_list:
                solver.add_deployment(**solver_kwargs)

            next_num_replicas, new_utility = solver.solve()

        return next_num_replicas, new_utility, solver_kwargs_list, next_num_input_list

    def autoscale(self, input_metrics, current_ts: float = None) -> List[Action]:
        if len(input_metrics) == 0:
            return []

        utilities = self.calculate_utilities(input_metrics)
        _LOGGER.info("[%.3f] sum: %.2f, current utilities: %s",
                     current_ts, sum(utilities.values()), str(utilities))

        sol = self.solve_optim(input_metrics, current_ts)
        if sol is None:
            return []

        next_num_replicas, new_utility, solver_kwargs_list, next_num_input_list = sol
        _LOGGER.info("[%.3f] new utility sum: %.2f, new_num_replicas: %s",
                     current_ts, np.sum(new_utility), next_num_replicas)

        has_none_next_inputs = [True for next_num_inputs in next_num_input_list
                                if next_num_inputs is None]
        if self.slack_resources is not None and len(has_none_next_inputs) < len(next_num_replicas):
            current_quantile = 0.0
            # store original quantiles to restore later
            quantiles = defaultdict(dict)
            for head_name, cluster_metrics in input_metrics.items():
                for deployment_name in cluster_metrics.keys():
                    quantile = self.predictors[head_name][deployment_name].quantile
                    quantiles[head_name][deployment_name] = quantile
                    current_quantile = max(quantile, current_quantile)

            new_sol = sol
            resource_per_replicas = [kwargs["resource_per_replica"]
                                     for kwargs in solver_kwargs_list]
            while current_quantile < 0.9:
                # check available resources w.r.t. the current deployment
                x = []
                for kwargs in solver_kwargs_list:
                    num_replicas = new_sol[0][kwargs["key"]]
                    if isinstance(num_replicas, tuple):
                        # has accept_rate:
                        x.append(num_replicas[0])
                    else:
                        x.append(num_replicas)
                used_resources = np.dot(resource_per_replicas, x)
                avail_resources = self.resource_limit["cpu"] - used_resources
                _LOGGER.info(
                    "[%.3f] solution num_replicas: %s used resources: %d avail resources: %d",
                    current_ts, x, used_resources, avail_resources)

                # if available resources are enough, break
                if avail_resources <= self.slack_resources["cpu"]:
                    break

                # update to the valid solution (new_sol)
                sol = new_sol
                # increment quantiles
                for head_name, cluster_metrics in input_metrics.items():
                    for deployment_name in cluster_metrics.keys():
                        predictor = self.predictors[head_name][deployment_name]
                        predictor.update_quantile(predictor.quantile + 0.1)
                        current_quantile = max(
                            predictor.quantile, current_quantile)
                # re-solve
                new_sol = self.solve_optim(input_metrics, current_ts)

            # update
            next_num_replicas, new_utility, _, _ = sol
            _LOGGER.info("[%.3f] final new utility sum: %.2f, new_num_replicas: %s",
                         current_ts, np.sum(new_utility), next_num_replicas)

            # restore quantiles
            for head_name, dep_quantiles in quantiles.items():
                for deployment_name, quantile in dep_quantiles.items():
                    self.predictors[head_name][deployment_name].update_quantile(
                        quantile)

        actions = []
        for (head_name, deployment_name), num_replicas in next_num_replicas.items():
            if isinstance(num_replicas, tuple):
                # has accept_rate
                num_replicas, accept_rate = num_replicas
                action = Action(head_name=head_name,
                                deployment_name=deployment_name,
                                target=num_replicas,
                                accept_rate=accept_rate)
            else:
                action = Action(head_name=head_name,
                                deployment_name=deployment_name,
                                target=num_replicas)
            actions.append(action)
        return actions


class OraclePredictionUtilityPolicy(OraclePredictionPolicy):

    def __init__(self, autoscale_config: AutoscaleConfig, input_path: str,
                 window_size_s: float, input_rate_bin_size_s: int,
                 aggregator: str = "avg", latency_estimator: str = "max",
                 use_current_queue_len: bool = False,
                 queue_len_limit: int = 400,
                 **kwargs):
        super().__init__(
            autoscale_config=autoscale_config, input_path=input_path,
            window_size_s=window_size_s, aggregator=aggregator,
            input_rate_bin_size_s=input_rate_bin_size_s,
            latency_estimator=latency_estimator,
            use_current_queue_len=use_current_queue_len,
            queue_len_limit=queue_len_limit)

        self.pred_utility_autoscaler = PredictionUtilityAutoscaler(
            predictors=self.predictors,
            calculate_estimate=self.calculate_estimate,
            queue_len_parser=self.queue_len_parser,
            window_size_s=window_size_s,
            input_rate_bin_size_s=input_rate_bin_size_s,
            autoscale_config=autoscale_config,
            **kwargs,
        )
        self.metric_parser = None

        _LOGGER.info("Oracle Pred utility policy: %s",
                     self.pred_utility_autoscaler.config_str())

    def autoscale(self, input_metrics, current_ts: float = None) -> List[Action]:
        return self.pred_utility_autoscaler.autoscale(input_metrics, current_ts)


class PTFPredictionUtilityPolicy(PyTorchForecastingPredictionPolicy):

    def __init__(self, autoscale_config: AutoscaleConfig, window_size_s: int,
                 input_rate_bin_size_s: int,
                 model_params: Dict[str, Dict[str, Dict]],
                 aggregator="avg", latency_estimator: str = "max",
                 use_current_queue_len: bool = False,
                 queue_len_limit: int = 400,
                 **kwargs):
        super().__init__(
            autoscale_config=autoscale_config, model_params=model_params,
            window_size_s=window_size_s, aggregator=aggregator,
            input_rate_bin_size_s=input_rate_bin_size_s,
            latency_estimator=latency_estimator,
            use_current_queue_len=use_current_queue_len,
            queue_len_limit=queue_len_limit)

        self.pred_utility_autoscaler = PredictionUtilityAutoscaler(
            predictors=self.predictors,
            calculate_estimate=self.calculate_estimate,
            queue_len_parser=self.queue_len_parser,
            window_size_s=window_size_s,
            input_rate_bin_size_s=input_rate_bin_size_s,
            autoscale_config=autoscale_config,
            **kwargs,
        )
        self.metric_parser = None

        _LOGGER.info("PTF Pred utility policy: %s",
                     self.pred_utility_autoscaler.config_str())

    def autoscale(self, input_metrics, current_ts: float = None) -> List[Action]:
        return self.pred_utility_autoscaler.autoscale(input_metrics, current_ts)


class DartsPredictionUtilityPolicy(DartsPredictionPolicy):

    def __init__(self, autoscale_config: AutoscaleConfig, window_size_s: int,
                 input_rate_bin_size_s: int,
                 model_params: Dict[str, Dict[str, Dict]],
                 aggregator="avg", latency_estimator: str = "max",
                 use_current_queue_len: bool = False,
                 queue_len_limit: int = 400,
                 **kwargs):
        super().__init__(
            autoscale_config=autoscale_config, model_params=model_params,
            window_size_s=window_size_s, aggregator=aggregator,
            input_rate_bin_size_s=input_rate_bin_size_s,
            latency_estimator=latency_estimator,
            use_current_queue_len=use_current_queue_len,
            queue_len_limit=queue_len_limit)

        self.pred_utility_autoscaler = PredictionUtilityAutoscaler(
            predictors=self.predictors,
            calculate_estimate=self.calculate_estimate,
            queue_len_parser=self.queue_len_parser,
            window_size_s=window_size_s,
            input_rate_bin_size_s=input_rate_bin_size_s,
            autoscale_config=autoscale_config,
            skip_without_pred=False,
            **kwargs,
        )
        self.metric_parser = None

        _LOGGER.info("Darts Pred utility policy: %s",
                     self.pred_utility_autoscaler.config_str())

    def autoscale(self, input_metrics, current_ts: float = None) -> List[Action]:
        return self.pred_utility_autoscaler.autoscale(input_metrics, current_ts)


class GluonTSPredictionPolicy(BasePredictionPolicy):

    def __init__(self, autoscale_config: AutoscaleConfig, window_size_s: int,
                 input_rate_bin_size_s: int,
                 model_params: Dict[str, Dict[str, Dict]],
                 aggregator: str = "avg", latency_estimator: str = "max",
                 use_current_queue_len: bool = False,
                 queue_len_limit: int = 400):
        self.model_params = model_params
        super().__init__(
            autoscale_config=autoscale_config, window_size_s=window_size_s,
            input_rate_bin_size_s=input_rate_bin_size_s, aggregator=aggregator,
            latency_estimator=latency_estimator,
            use_current_queue_len=use_current_queue_len,
            queue_len_limit=queue_len_limit)

    def build_predictors(self) -> DefaultDict[str, Dict[str, Predictor]]:
        from gluonts_predictor import GluonTSPredictor
        predictors = defaultdict(dict)
        for cluster_name, jobs in self.model_params.items():
            for job_name, model_param in jobs.items():
                head_name = cluster_name + "-ray-head"
                predictors[head_name][job_name] = \
                    GluonTSPredictor(window_size_s=self.window_size_s,
                                     head_name=head_name,
                                     deployment_name=job_name,
                                     unit_time_s=self.input_rate_bin_size_s,
                                     **model_param)
        return predictors


class GluonTSPredictionUtilityPolicy(GluonTSPredictionPolicy):

    def __init__(self, autoscale_config: AutoscaleConfig, window_size_s: int,
                 input_rate_bin_size_s: int,
                 model_params: Dict[str, Dict[str, Dict]],
                 aggregator="avg", latency_estimator: str = "max",
                 use_current_queue_len: bool = False,
                 queue_len_limit: int = 400,
                 **kwargs):
        super().__init__(
            autoscale_config=autoscale_config, model_params=model_params,
            window_size_s=window_size_s, aggregator=aggregator,
            input_rate_bin_size_s=input_rate_bin_size_s,
            latency_estimator=latency_estimator,
            use_current_queue_len=use_current_queue_len,
            queue_len_limit=queue_len_limit)

        self.pred_utility_autoscaler = PredictionUtilityAutoscaler(
            predictors=self.predictors,
            calculate_estimate=self.calculate_estimate,
            queue_len_parser=self.queue_len_parser,
            window_size_s=window_size_s,
            input_rate_bin_size_s=input_rate_bin_size_s,
            autoscale_config=autoscale_config,
            skip_without_pred=False,
            **kwargs,
        )
        self.metric_parser = None

        _LOGGER.info("GluonTS Pred utility policy: %s",
                     self.pred_utility_autoscaler.config_str())

    def autoscale(self, input_metrics, current_ts: float = None) -> List[Action]:
        return self.pred_utility_autoscaler.autoscale(input_metrics, current_ts)


_POLICY_CLASSES = {
    "oneshot": OneshotPolicy,
    "oneshot_ema": OneshotEMAPolicy,
    "aiad": AIADPolicy,
    "ai": AIPolicy,
    "omiomd": OMIOMDPolicy,
    "none": NonePolicy,
    "pred_oracle": OraclePredictionPolicy,
    "pred_pft": PyTorchForecastingPredictionPolicy,
    "pred_darts": DartsPredictionPolicy,
    "utility": UtilityPolicy,
    "pred_oracle_utility": OraclePredictionUtilityPolicy,
    "pred_pft_utility": PTFPredictionUtilityPolicy,
    "pred_darts_utility": DartsPredictionUtilityPolicy,
    "pred_gluonts": GluonTSPredictionPolicy,
    "pred_gluonts_utility": GluonTSPredictionUtilityPolicy,
}


def create_policy(policy_name: str, autoscale_config: AutoscaleConfig,
                  *args, **kwargs) -> Policy:
    return _POLICY_CLASSES[policy_name](autoscale_config, *args, **kwargs)
