import kubernetes
import kubernetes.client
from kubernetes.config.config_exception import ConfigException


__configured = False
__core_api = None
__custom_objects_api = None


def __load_config():
    global __configured
    if __configured:
        return
    try:
        kubernetes.config.load_incluster_config()
    except ConfigException:
        kubernetes.config.load_kube_config()
    __configured = True


def core_api() -> kubernetes.client.CoreV1Api:
    global __core_api
    if __core_api is None:
        __load_config()
        __core_api = kubernetes.client.CoreV1Api()

    return __core_api


def custom_objects_api() -> kubernetes.client.CustomObjectsApi:
    global __custom_objects_api
    if __custom_objects_api is None:
        __load_config()
        __custom_objects_api = kubernetes.client.CustomObjectsApi()

    return __custom_objects_api
