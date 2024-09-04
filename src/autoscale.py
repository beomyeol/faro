from abc import abstractmethod
from typing import Dict
from dataclasses import dataclass
from decimal import Decimal
from statistics import mean
from typing import Any

from ray.serve._private.custom_autoscaling import (
    LATENCY_KEY,
    QUEUE_LEN_KEY,
    HEAD_QUEUE_LEN_KEY,
    INPUT_RATE_KEY,
    DROP_RATE_KEY,
)

from resources import Resources


@dataclass
class Action:
    head_name: str
    deployment_name: str
    factor: float = None
    delta: int = None
    limit: int = None
    target: int = None
    resources: Resources = Resources(cpu=Decimal(1))
    trials: int = 1
    accept_rate: int = None


@dataclass
class ClusterConfig:
    upscale_delay_s: float = 30.0
    downscale_delay_s: float = 600.0
    target_metric: float = 800
    min_replicas: int = 1
    max_replicas: int = None


@dataclass
class AutoscaleConfig:
    interval_s: float = 5.0
    upscale_delay_s: float = 30.0
    downscale_delay_s: float = 600.0
    metric: str = "avg_latency"
    target_metric: float = 800
    min_replicas: int = 1
    max_replicas: int = None
    cluster_configs: Dict[str, ClusterConfig] = None
    look_back_period_s: float = 30.0
    max_trials: int = 1
    backoff_s: float = 10.0
    downscale_action_delay_s: int = 0

    @property
    def upscale_threshold(self):
        return self.upscale_delay_s // self.interval_s

    @property
    def downscale_threshold(self):
        return self.downscale_delay_s // self.interval_s

    @staticmethod
    def create(**kwargs):
        retval = AutoscaleConfig(**kwargs)
        if retval.cluster_configs is not None:
            new_cluster_configs = {}
            for head_name, cluster_config in retval.cluster_configs.items():
                new_cluster_configs[head_name] = ClusterConfig(
                    **cluster_config)
            retval.cluster_configs = new_cluster_configs
        return retval


class MetricParser:

    @abstractmethod
    def __call__(self, metrics: Dict[str, Any]) -> Any:
        pass


class AvgLatencyMetricParser(MetricParser):

    def __call__(self, metrics: Dict[str, Any]):
        if LATENCY_KEY not in metrics or metrics[LATENCY_KEY] is None:
            # when there is no request
            return 0
        else:
            return metrics[LATENCY_KEY]


class AvgQueueLenMetricParser(MetricParser):

    def __call__(self, metrics: Dict[str, Any]):
        return mean(metrics[QUEUE_LEN_KEY])


class TotalQueueLenMetricParser(MetricParser):

    def __call__(self, metrics: Dict[str, Any]):
        return sum(metrics[QUEUE_LEN_KEY]) + metrics[HEAD_QUEUE_LEN_KEY]


class InputRateMetricParser(MetricParser):

    def __call__(self, metrics: Dict[str, Any]):
        return metrics.get(INPUT_RATE_KEY, None)


class DropRateMetricParser(MetricParser):

    def __call__(self, metrics: Dict[str, Any]):
        return metrics.get(DROP_RATE_KEY, None)


_METRIC_PARSER = {
    "avg_latency": AvgLatencyMetricParser,
    "avg_queue_len": AvgQueueLenMetricParser,
    "total_queue_len": TotalQueueLenMetricParser,
    "input_rate": InputRateMetricParser,
    "drop_rate": DropRateMetricParser,
}


def create_metric_parser(name: str) -> MetricParser:
    return _METRIC_PARSER[name]()
