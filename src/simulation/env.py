import simpy

__global_env = None


def global_env() -> simpy.Environment:
    global __global_env
    if __global_env is None:
        __global_env = simpy.Environment()
    return __global_env
