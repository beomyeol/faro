from collections import defaultdict
import logging
from typing import Optional

import kopf

from server import WebhookServiceServer
from quota_manager import RayClusterQuotaManager
from resources import Resources
from k8s_utils import get_ray_cluster_utilization
from kubernetes.utils import parse_quantity

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.DEBUG)
_MANAGER = RayClusterQuotaManager()


@kopf.on.validate("pod", id="validate-ray-create", operation="CREATE",
                  labels={"ray-cluster-name": kopf.PRESENT})
def validate_ray_create(namespace, labels, spec, **kwargs):
    if not _MANAGER.handle_create(labels, spec, namespace):
        raise kopf.AdmissionError("denied", code=403)


@kopf.on.create("clusterquotas")
@kopf.on.update("clusterquotas")
@kopf.on.resume("clusterquotas")
def create_or_update_cluster_quota(body, **_):
    namespace = body["metadata"]["namespace"]
    spec = body["spec"]
    cluster_name = spec["clusterName"]
    resources = spec["resources"]
    _MANAGER.update_cluster_quota(cluster_name, resources, namespace)


@kopf.on.startup()
def configure(settings: kopf.OperatorSettings, **_):
    settings.admission.server = WebhookServiceServer(
        namespace="k8s-ray", name="webhook-service",
        addr="0.0.0.0", port=9443)
    settings.admission.managed = "auto.kopf.dev"


_CPU_UTIL_UP_THRESHOLD = 0.8
_MEM_UTIL_UP_THRESHOLD = 0.8
_CPU_UTIL_DOWN_THREADHOLD = 0.2
_MONITOR_INTERVAL_S = 1
_METRICS_SERVER_RESOLUTION = 15
_UPSCALE_COUNT = 30 // _METRICS_SERVER_RESOLUTION  # 30s / 15s
_DOWNSCALE_COUNT = 600 // _METRICS_SERVER_RESOLUTION  # 600s / 15s
# _UPSCALE_COUNT = 15 // _METRICS_SERVER_RESOLUTION  # 15s / 15s
# _DOWNSCALE_COUNT = 300 // _METRICS_SERVER_RESOLUTION  # 300 / 15s
_UNIT_RESOURCES = Resources(cpu=parse_quantity("1"),
                            memory=parse_quantity("4Gi"))


class K8SOnlySimplePolicy:

    def __init__(self, stopped, cluster_name: str, namespace: str,
                 interval_s: float):
        self.stopped = stopped
        self.cluster_name = cluster_name
        self.namespace = namespace
        self.interval_s = interval_s

        # Use this to check whether a new read happens or not.
        # The k8s metric server resolution is _METRICS_SERVER_RESOLUTION.
        self.prev_utilization = None

        self.upscale_count = 0
        self.downscale_count = 0

    def get_utilization(self) -> Optional[dict]:
        """Returns utilization information for the cluster pods.

        This returns None if there is no change since the previous read.
        """
        utilization = get_ray_cluster_utilization(
            self.cluster_name, self.namespace)
        if self.prev_utilization is None:
            self.prev_utilization = utilization
            return utilization

        # check whether this is the same as the previous read
        if self.prev_utilization == utilization:
            return None
        else:
            self.prev_utilization = utilization
            return utilization

    def step(self) -> None:
        utilization_dict = self.get_utilization()
        if utilization_dict is None:
            # no update from the previous read.
            return

        # compute average utilization
        avg_utilization = Resources()
        utilization_count = 0
        for pod_name, info in utilization_dict.items():
            # only see workers for now since the added quota will used for
            # workers. not the head.
            if info["type"] == "head":
                continue
            for container in info["containers"]:
                utilization = container["utilization"]
                _LOGGER.debug("Utilization for pod %s: %s",
                              pod_name, utilization)
                avg_utilization += utilization
                utilization_count += 1

        if utilization_count == 0:
            return

        avg_utilization.cpu /= utilization_count
        avg_utilization.memory /= utilization_count

        _LOGGER.debug("Avg utilization for cluster %s: %s",
                      self.cluster_name, avg_utilization)
        if avg_utilization.cpu >= _CPU_UTIL_UP_THRESHOLD:
            # upscale
            self.downscale_count = 0
            self.upscale_count += 1
            _LOGGER.debug("Increment upscale count for %s: %d",
                          self.cluster_name, self.upscale_count)
            if self.upscale_count >= _UPSCALE_COUNT:
                self.upscale_count = 0
                # TODO: check max, otherwise this increases indefinitely
                _LOGGER.info("Upscale cluster=%s", self.cluster_name)
                _MANAGER.increase_quota(self.cluster_name, _UNIT_RESOURCES)
        elif avg_utilization.cpu < _CPU_UTIL_DOWN_THREADHOLD:
            # downscale
            self.upscale_count = 0
            self.downscale_count += 1
            _LOGGER.debug("Increment downscale count for %s: %d",
                          self.cluster_name, self.downscale_count)
            if self.downscale_count >= _DOWNSCALE_COUNT:
                _LOGGER.info("Downscale cluster=%s", self.cluster_name)
                self.downscale_count = 0
                if len(utilization_dict) > 2:
                    # assume that there is a single head pod and
                    # others are worker pods
                    # 1 head pod + at least 1 worker pod should run
                    # TODO: configure this value for each cluster
                    _MANAGER.increase_quota(self.cluster_name,
                                            -_UNIT_RESOURCES)
                else:
                    _LOGGER.debug(
                        "No downscale cluster=%s. num_pods: %d <= 2",
                        self.cluster_name, len(utilization_dict))
        else:
            self.upscale_count = 0
            self.downscale_count = 0

    def run(self):
        _LOGGER.info("monitor starts for %s (ns=%s)",
                     self.cluster_name, self.namespace)
        while not self.stopped:
            self.step()
            self.stopped.wait(self.interval_s)
        _LOGGER.info("monitor done for %s (ns=%s)",
                     self.cluster_name, self.namespace)


@kopf.daemon("clusterquotas")
def monitor_cluster(stopped, body, **kwargs):
    namespace = body["metadata"]["namespace"]
    spec = body["spec"]
    cluster_name = spec["clusterName"]

    policy = K8SOnlySimplePolicy(stopped, cluster_name, namespace,
                                 interval_s=_MONITOR_INTERVAL_S)
    policy.run()
