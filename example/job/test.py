import os
import ray
import time
import numpy as np

@ray.remote
def estimate_pi(num_samples):
    time.sleep(6)  # trigger autoscaler since default autoscaler update period is 5 seconds
    xs = np.random.uniform(low=-1.0, high=1.0, size=num_samples)
    ys = np.random.uniform(low=-1.0, high=1.0, size=num_samples)
    xys = np.stack((xs, ys), axis=-1)
    inside = xs*xs + ys*ys <= 1.0
    xys_inside = xys[inside]
    in_circle = xys_inside.shape[0]
    approx_pi = 4.0*in_circle/num_samples
    return approx_pi

if __name__ == "__main__":
    ray.init("ray://example-cluster-ray-head:10001")
    
    num_samples = 10000
    num_tasks = 4
    refs = [estimate_pi.remote(num_samples) for _ in range(num_tasks)]
    ray.get(refs)