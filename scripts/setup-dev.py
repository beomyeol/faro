import argparse
from pathlib import Path
import subprocess

import ray

if __name__ == "__main__":
    package_ray_dir = Path(ray.__file__).parent
    print(f"package ray dir: {package_ray_dir.as_posix()}")

    package_serve_dir = package_ray_dir.joinpath("serve")
    print(f"package ray serve dir: {package_serve_dir.as_posix()}")
    local_ray_dir = Path(__file__).parents[1].joinpath("ray")
    local_serve_dir = local_ray_dir.joinpath("serve").resolve()
    print(f"local ray serve dir: {local_serve_dir.as_posix()}")
    assert local_serve_dir.exists()

    for local in local_serve_dir.glob("**/*.py"):
        target = package_serve_dir.joinpath(local.relative_to(local_serve_dir))
        print(f"Creating symbolic link from \n {local} to \n {target}")
        subprocess.check_call(["rm", "-rf", target])
        subprocess.check_call(["ln", "-s", local, target])

    package_ray_private_dir = package_ray_dir.joinpath("_private")
    print(f"package ray private dir: {package_ray_private_dir.as_posix()}")
    local_private_dir = local_ray_dir.joinpath("_private").resolve()
    print(f"local ray private dir: {local_private_dir.as_posix()}")
    assert local_private_dir.exists()

    for local in local_private_dir.glob("**/*.py"):
        target = package_ray_private_dir.joinpath(local.relative_to(local_private_dir))
        print(f"Creating symbolic link from \n {local} to \n {target}")
        subprocess.check_call(["rm", "-rf", target])
        subprocess.check_call(["ln", "-s", local, target])

    package_ray_autoscaler_dir = package_ray_dir.joinpath("autoscaler")
    print(f"package ray autoscaler dir: {package_ray_autoscaler_dir.as_posix()}")
    local_autoscaler_dir = local_ray_dir.joinpath("autoscaler").resolve()
    print(f"local ray autoscaler dir: {local_autoscaler_dir.as_posix()}")
    assert local_autoscaler_dir.exists()

    for local in local_autoscaler_dir.glob("**/*.py"):
        target = package_ray_autoscaler_dir.joinpath(local.relative_to(local_autoscaler_dir))
        print(f"Creating symbolic link from \n {local} to \n {target}")
        subprocess.check_call(["rm", "-rf", target])
        subprocess.check_call(["ln", "-s", local, target])