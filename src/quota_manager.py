import logging
from k8s import core_api
from resources import Resources
from k8s_utils import *
from kubernetes.client.exceptions import ApiException
import kopf

_LOGGER = logging.getLogger(__name__)


class RayClusterQuotaManager:

    def get_cluster_quota(self, cluster_name) -> Resources:
        try:
            return parse_resources(
                get_ray_clusterquota(cluster_name)["spec"]["resources"])
        except ApiException as e:
            _LOGGER.warning(f"Exception in getting cluster quota:\n{e}")
            return None

    def update_cluster_quota(self, cluster_name, resources, namespace) -> None:
        _LOGGER.info(f"update cluster quota. cluster_name={cluster_name}, "
                     f"quota={resources}, namespace={namespace}")
        quota = parse_resources(resources)
        pods = get_ray_cluster_pods(cluster_name, namespace)
        used_resources = Resources()
        for pod in pods:
            used_resources += get_pod_requested_resources(pod)

        while (used_resources.cpu > quota.cpu
                or used_resources.memory > quota.memory):
            _LOGGER.info(
                f"Quota exceeded. quota={quota}, used={used_resources}")
            for pod in pods:
                node_type = pod.metadata.labels["ray-node-type"]
                if node_type == "head":
                    continue
                # TODO: kill workers more intelligently
                name = pod.metadata.name
                namespace = pod.metadata.namespace
                _LOGGER.info(
                    f"Killing a pod name={name}, namespace={namespace}")
                try:
                    core_api().delete_namespaced_pod(name, namespace)
                    used_resources -= get_pod_requested_resources(pod)
                    if (used_resources.cpu <= quota.cpu
                            and used_resources.memory <= quota.memory):
                        break
                except ApiException as e:
                    raise kopf.TemporaryError(
                        f"Exception in killing a worker pod:\n{e}",
                        delay=30)

    def handle_create(self, labels, spec, namespace) -> bool:
        cluster_name = labels["ray-cluster-name"]
        node_type = labels["ray-user-node-type"]
        resources_list = [container["resources"]["requests"]
                          for container in spec["containers"]]
        requested_resources = Resources()
        for resources in resources_list:
            requested_resources += parse_resources(resources)
        quota = self.get_cluster_quota(cluster_name)
        _LOGGER.debug(
            f"CREATE cluster_name={cluster_name}, type={node_type}, "
            f"requested={requested_resources}, quota={quota}")

        if quota is None:
            _LOGGER.warning(f"No quota exists for {cluster_name}")
            return True

        used_resources = get_ray_cluster_requested_resources(
            cluster_name, namespace)
        _LOGGER.debug(f"used_resources={used_resources}")

        new_resources = used_resources + requested_resources
        if (new_resources.cpu > quota.cpu or
                new_resources.memory > quota.memory):
            _LOGGER.info(
                f"Quota exceeded. quota={quota}, "
                f"used={used_resources}, requested={requested_resources}")
            return False

        return True

    def increase_quota(
            self, cluster_name: str, resource_delta: Resources) -> None:
        current_resources = self.get_cluster_quota(cluster_name)
        new_resources = current_resources + resource_delta
        _LOGGER.info("Change quota for '%s' from %s to %s",
                     cluster_name, current_resources, new_resources)
        patch_ray_clusterquota(cluster_name, new_resources)
