import logging
from pathlib import Path
import sys

_LOGGER = logging.getLogger(__name__)


def main(root_dir):
    # update src files
    controller_pod = k8s_utils.list_autoscale_controller_pod()[0]
    pod_name = controller_pod.metadata.name
    _LOGGER.info(f"controller pod: {pod_name}")
    k8s_utils.exec_pod_cmd(pod_name, "sudo rm -r /home/ray/src")
    k8s_utils.copy_to_pod(
        root_dir.joinpath("src").as_posix(), pod_name, "/home/ray")


if __name__ == "__main__":
    format = '%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s'
    logging.basicConfig(level=logging.INFO, format=format)
    root_dir = Path(__file__).resolve().parents[1]
    src_dir = root_dir.joinpath("src").as_posix()
    sys.path.append(src_dir)
    from configuration import load_config
    import k8s_utils

    main(root_dir)
