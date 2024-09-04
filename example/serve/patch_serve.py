from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging

import k8s_utils

_LOGGER = logging.getLogger(__name__)


def patch_serve():
    root_dir = Path(__file__).resolve().parents[2]
    local_serve_dir = root_dir.joinpath("ray").resolve()
    _LOGGER.info("local ray serve dir: %s", local_serve_dir)
    assert local_serve_dir.exists()

    pod_serve_dir = Path(
        "/home/ray/anaconda3/lib/python3.7/site-packages/ray/serve")

    head_pods = k8s_utils.get_head_pods()

    with ThreadPoolExecutor() as executor:
        for head_pod in head_pods:
            for local in local_serve_dir.glob("**/*.py"):
                local_path = local.resolve()
                executor.submit(
                    k8s_utils.copy_to_pod,
                    local_path=local_path,
                    pod_name=head_pod,
                    target_path=pod_serve_dir.joinpath(
                        local.relative_to(local_serve_dir))
                )


if __name__ == "__main__":
    format = "[%(asctime)s] %(name)s:%(lineno)d [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO,
                        format=format)
    patch_serve()
