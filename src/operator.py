import kopf
import logging

from server import WebhookServiceServer
from quota_manager import RayClusterQuotaManager

_LOGGER = logging.getLogger(__name__)
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
