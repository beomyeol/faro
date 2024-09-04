from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

from resources import Resources, parse_resource_dict
from simulation import metric

from simulation.env import global_env


class Task:

    def __init__(self, cluster_name) -> None:
        self.cluster_name = cluster_name
        self.arrival_ts = None
        self.start_ts = None
        self.end_ts = None
        self.done_event = global_env().event()
        self.succeeded = None

    def arrive(self):
        self.arrival_ts = global_env().now

    def start(self):
        self.start_ts = global_env().now

    def done(self, succeeded: bool = True):
        if self.succeeded is None:
            # for failed tasks, this can be set already
            self.succeeded = succeeded
        self.end_ts = global_env().now
        metric.put(self)
        self.done_event.succeed()

    def latency(self):
        return self.end_ts - self.start_ts

    def wait_time(self):
        return self.start_ts - self.arrival_ts

    def elapsed_time(self):
        return self.end_ts - self.arrival_ts

    def stats(self):
        return (f"arrival={self.arrival_ts:.3f}, start={self.start_ts:.3f}, "
                f"end={self.end_ts}")


@dataclass
class Pod:
    node_name: str
    index: int
    resources: Resources

    def __repr__(self):
        return f"Pod({self.node_name}, {self.index}, {self.resources})"


class Job(ABC):

    def __init__(self, resources: Resources) -> None:
        self._resources = resources
        self._worker: Worker = None

    @property
    def resources(self):
        return self._resources

    def set_worker(self, worker: Worker):
        self._worker = worker
        worker.allocate(self)

    def unset_worker(self):
        self._worker.deallocate(self)
        self._worker = None

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def run(self):
        pass


class Worker:

    def __init__(self, pod: Pod) -> None:
        self.pod = pod
        self.active_jobs: Dict[int, Job] = {}
        self._idle_start_ts: float = global_env().now

    @property
    def resources(self):
        return self.pod.resources

    def get_idle_time(self) -> float:
        if self._idle_start_ts is None:
            return 0.0
        else:
            return global_env().now - self._idle_start_ts

    def allocate(self, job: Job) -> None:
        if job.resources > self.available_resources():
            raise ValueError("cannot allocate job. no available resources")
        self.active_jobs[id(job)] = job
        self._idle_start_ts = None

    def deallocate(self, job: Job) -> None:
        del self.active_jobs[id(job)]
        if len(self.active_jobs) == 0:
            self._idle_start_ts = global_env().now

    def used_resources(self) -> Resources:
        total = Resources()
        for job in self.active_jobs.values():
            total += job.resources
        return total

    def available_resources(self) -> Resources:
        return self.resources - self.used_resources()

    def stop(self):
        for job in self.active_jobs.values():
            job.stop()


class Node:

    def __init__(self, name: str, resources: Resources):
        self.name: str = name
        self.resources = resources
        self.pods: List[Pod] = []
        self.pod_index = 0

        self.used_resources = Resources()

    @property
    def available_resources(self):
        return self.resources - self.used_resources

    def create_pod(self, request: Resources) -> Pod:
        avail = self.available_resources
        if avail < request:
            raise ValueError(f"requested={request}, avail={avail}")
        self.used_resources += request
        pod = Pod(self.name, self.pod_index, request)
        self.pod_index += 1
        self.pods.append(pod)
        return pod

    def remove_pod(self, pod: Pod) -> None:
        self.used_resources -= pod.resources
        self.pods.remove(pod)
