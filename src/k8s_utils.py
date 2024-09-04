import subprocess
from typing import Dict, List
from collections import defaultdict
import logging
from decimal import Decimal, ROUND_CEILING

from kubernetes.utils import parse_quantity
from kubernetes.client import V1PodList, V1Pod

from k8s import core_api, custom_objects_api
from resources import Resources, parse_resources

_LOGGER = logging.getLogger(__name__)


def get_nodes_allocatable_resources() -> Dict[str, Resources]:
    # assume pods are always allocatable.
    retval = {}
    for node in core_api().list_node().items:
        allocatable = node.status.allocatable
        retval[node.metadata.name] = Resources(
            cpu=parse_quantity(allocatable["cpu"]),
            memory=parse_quantity(allocatable["memory"]))
    return retval


def get_available_resource_quota(namespace: str = "k8s-ray", name="k8s-ray-resources", factor=1):
    quotas = core_api().list_namespaced_resource_quota(namespace)
    retval = None
    for quota in quotas.items:
        if quota.metadata.name != name:
            continue
        retval = Resources(
            cpu=(parse_quantity(quota.status.hard["requests.cpu"]) -
                 parse_quantity(quota.status.used["requests.cpu"])),
            memory=(parse_quantity(quota.status.hard["requests.memory"]) -
                    parse_quantity(quota.status.used["requests.memory"])))
    if factor != 1:
        # Use interger to avoid floating point issue....
        v_cpu = (retval.cpu * Decimal(factor)
                 ).to_integral_exact(rounding=ROUND_CEILING)
        _LOGGER.info("Converting vCPU %s to %s", retval.cpu, v_cpu)
        retval.cpu = v_cpu
    return retval


def get_non_terminated_pods(namespace: str = None) -> List[V1Pod]:
    field_selector = ",".join(
        [
            "status.phase!=Failed",
            "status.phase!=Unknown",
            "status.phase!=Succeeded",
            "status.phase!=Terminating",
        ]
    )
    if namespace is None:
        pod_list = core_api().list_pod_for_all_namespaces(
            field_selector=field_selector)
    else:
        pod_list = core_api().list_namespaced_pod(
            namespace=namespace,
            field_selector=field_selector)
    # Don't return pods marked for deletion,
    # i.e. pods with non-null metadata.DeletionTimestamp.
    return [
        pod
        for pod in pod_list.items
        if pod.metadata.deletion_timestamp is None
    ]


def get_nodes_resources_usages() -> Dict[str, Resources]:
    pod_list = get_non_terminated_pods()
    retval = defaultdict(Resources)
    for pod in pod_list:
        node = pod.spec.node_name
        for container in pod.spec.containers:
            requests = container.resources.requests
            if requests is not None:
                retval[node] += Resources(
                    parse_quantity(requests.get("cpu", 0)),
                    parse_quantity(requests.get("memory", 0)))
    return retval


def get_nodes_resource_utilization() -> Dict[str, Resources]:
    allocable_resources = get_nodes_allocatable_resources()
    used_resources = get_nodes_resources_usages()
    retval = {}
    for node, allocatable in allocable_resources.items():
        used = used_resources[node]
        retval[node] = used / allocatable
    return retval


def get_nodes_free_resources() -> Dict[str, Resources]:
    allocatable_resources = get_nodes_allocatable_resources()
    used_resources = get_nodes_resources_usages()

    for node, used_resources in used_resources.items():
        if node is None:
            # pending pods
            _LOGGER.info("Pending resources: %s", used_resources)
            continue
        allocatable_resources[node] -= used_resources
    return allocatable_resources


def get_ray_cluster_pods(cluster_name=None, namespace="k8s-ray",
                         field_selector=None) -> V1PodList:
    label_selector = "ray-cluster-name"
    if cluster_name is not None:
        label_selector += f"={cluster_name}"
    return core_api().list_namespaced_pod(
        namespace, label_selector=label_selector,
        field_selector=field_selector).items


_QUOTA_GROUP = "ray.beomyeol.github.io"
_QUOTA_VERSION = "v1"
_QUOTA_PLURAL = "clusterquotas"


def get_ray_clusterquota(cluster_name: str, namespace="k8s-ray") -> dict:
    return custom_objects_api().get_namespaced_custom_object(
        _QUOTA_GROUP, _QUOTA_VERSION, namespace, _QUOTA_PLURAL,
        f"{cluster_name}-quota")


def patch_ray_clusterquota(cluster_name: str, resources: Resources,
                           namespace="k8s-ray"):
    body = {"spec": {"resources": {"cpu": int(resources.cpu),
                                   "memory": int(resources.memory)}}}
    return custom_objects_api().patch_namespaced_custom_object(
        _QUOTA_GROUP, _QUOTA_VERSION, namespace, _QUOTA_PLURAL,
        f"{cluster_name}-quota", body)


def get_pod_requested_resources(pod) -> Resources:
    used_resources = Resources()
    for container in pod.spec.containers:
        used_resources += parse_resources(container.resources.requests)
    return used_resources


def get_ray_cluster_requested_resources(cluster_name,
                                        namespace="k8s-ray") -> Resources:
    requested_resources = Resources()
    pods = get_ray_cluster_pods(cluster_name, namespace)
    for pod in pods:
        requested_resources += get_pod_requested_resources(pod)
    return requested_resources


def get_ray_cluster_utilization(cluster_name, namespace="k8s-ray") -> dict:
    outs = {}
    pod_metrics = custom_objects_api().list_namespaced_custom_object(
        "metrics.k8s.io", "v1beta1",
        "k8s-ray", "pods",
        label_selector=f"ray-cluster-name={cluster_name}")["items"]

    for pod_metric in pod_metrics:
        pod_name = pod_metric["metadata"]["name"]
        parsed_containers = []
        for container in pod_metric["containers"]:
            parsed_containers.append({
                "name": container["name"],
                "usage": parse_resources(container["usage"]),
            })
        outs[pod_name] = {"containers": parsed_containers}

    # get requsted resource information
    for pod in get_ray_cluster_pods(cluster_name, namespace=namespace):
        pod_name = pod.metadata.name
        if pod_name not in outs:
            # This can happen when a new pod is created between the metric
            # call and list pod.
            # Skip this pod.
            continue
        value = outs[pod_name]
        value["type"] = pod.metadata.labels["ray-node-type"]
        for out_container, container in zip(value["containers"], pod.spec.containers):
            if out_container["name"] != container.name:
                raise ValueError('Different name. {} != {}'.format(
                    out_container["name"], container.name))
            requests = parse_resources(container.resources.requests)
            out_container["requests"] = requests
            # utilization
            usage = out_container["usage"]
            out_container["utilization"] = usage / requests

    return outs


_RAY_HEAD_SUFFIX = "-ray-head"


def list_ray_head_cluster_ips(namespace="k8s-ray"):
    cluster_ips = {}
    service_list = core_api().list_namespaced_service(namespace)
    for service in service_list.items:
        name = service.metadata.name
        if not name.endswith(_RAY_HEAD_SUFFIX):
            continue
        name = name[:-len(_RAY_HEAD_SUFFIX)]
        cluster_ips[name] = service.spec.cluster_ip
    return cluster_ips


def copy_to_pod(local_path: str, pod_name: str, target_path: str):
    pod_path = f"{pod_name}:{target_path}"
    _LOGGER.info("Copying %s to %s...", local_path, pod_path)
    subprocess.check_call([
        "kubectl", "cp", "-n", "k8s-ray", str(local_path), pod_path
    ])


def copy_from_pod(pod_name: str, target_path: str, local_path):
    pod_path = f"{pod_name}:{target_path}"
    _LOGGER.info("Copying %s to %s...", pod_path, str(local_path))
    subprocess.check_call([
        "kubectl", "cp", "-n", "k8s-ray", pod_path, str(local_path)
    ])


def list_autoscale_controller_pod(namespace="k8s-ray"):
    label_selector = "application=faro-operator"
    return core_api().list_namespaced_pod(
        namespace, label_selector=label_selector).items


def list_ray_head_pods(namespace="k8s-ray"):
    label_selector = "ray-user-node-type=head-node"
    head_pods = core_api().list_namespaced_pod(
        namespace, label_selector=label_selector).items
    return {pod.metadata.labels["ray-cluster-name"]: pod.metadata.name
            for pod in head_pods}


def list_ray_worker_pods(namespace="k8s-ray"):
    label_selector = "ray-user-node-type=worker-node"
    worker_pods = core_api().list_namespaced_pod(
        namespace, label_selector=label_selector).items
    retval = defaultdict(list)
    for pod in worker_pods:
        retval[pod.metadata.labels["ray-cluster-name"]
               ].append(pod.metadata.name)
    return retval


def exec_pod_cmd(pod_name, cmd):
    cmds = [
        "kubectl", "exec", "-n", "k8s-ray", f"pod/{pod_name}", "--", cmd,
    ]
    subprocess.run(" ".join(cmds), shell=True)
