from __future__ import annotations
import logging
from typing import List, Optional, Dict, TYPE_CHECKING, Tuple, Union
from dataclasses import dataclass

import simpy
from simulation.distribution import Distribution

from simulation.env import global_env
from simulation.primitives import Worker, Resources, Task
from simulation.inference import InferenceModel, InferenceTask, InferenceModelSpec

if TYPE_CHECKING:
    from controller import Controller

_LOGGER = logging.getLogger(__name__)


@dataclass
class ClusterSpec:
    name: str
    quota: Optional[Resources]
    worker_spec: Resources
    min_workers: int = 1
    max_workers: Optional[int] = None
    monitor_interval_s: float = 5
    idle_timeout_s: Optional[float] = None
    route_latency: Distribution = None


class InferenceTaskWrapper:

    def __init__(self, task: InferenceTask, callback):
        self.task = task
        self.callback = callback

    @property
    def start_ts(self):
        return self.task.start_ts

    def arrive(self):
        self.task.arrive()

    def start(self):
        self.task.start()

    def done(self, succeeded: bool = True):
        self.task.succeeded = succeeded
        self.callback(self)

    def stats(self):
        return self.task.stats()


class Cluster:

    def __init__(self, spec: ClusterSpec, controller: Controller) -> None:
        self.env = global_env()
        self.spec = spec
        self.controller = controller
        self.workers: List[Worker] = []
        self.queue = simpy.Store(self.env)
        self.proc: simpy.Process = self.env.process(
            self.run(self.queue, self.process))
        self.monitor_proc: simpy.Process = None

        self.models: Dict[str, InferenceModel] = {}

        _LOGGER.info("%s", spec)

    def initialize(self):
        for _ in range(self.spec.min_workers):
            if not self.add_worker():
                raise RuntimeError("Failed to initialize a cluster")
        if self.spec.idle_timeout_s is not None:
            self.monitor_proc = self.env.process(self.monitor())

    def monitor(self):
        _LOGGER.info(*self.log("cluster monitor gets started"))
        while True:
            try:
                yield self.env.timeout(self.spec.monitor_interval_s)
                candidates: List[Worker] = []
                for worker in self.workers:
                    if worker.get_idle_time() >= self.spec.idle_timeout_s:
                        candidates.append(worker)
                for candidate in candidates:
                    if len(self.workers) > self.spec.min_workers:
                        _LOGGER.info(*self.log(
                            "removing idle worker... idle_time=%.2f",
                            candidate.get_idle_time()
                        ))
                        self.remove_worker(candidate)
            except simpy.Interrupt:
                break
        _LOGGER.info(*self.log("cluster monitor stopped"))

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def max_workers(self) -> Optional[int]:
        return self.spec.max_workers

    def log(self, msg, *args):
        new_msg = "[%.3f] (%s) " + msg
        new_args = (self.env.now, self.name, *args)
        return (new_msg, *new_args)

    def available_resources(self) -> List[Resources]:
        return [worker.available_resources()
                for worker in self.workers]

    def used_resources(self) -> Resources:
        total_resources = Resources()
        for worker in self.workers:
            total_resources += worker.used_resources()
        return total_resources

    def submit(self, task: Task):
        task.arrive()
        if isinstance(task, InferenceTask):
            # simulate sequential routing at http proxy
            self.queue.put(task)
        else:
            raise NotImplementedError()

    def process(self, task):
        # self.models[task.model].handle(task)
        if isinstance(task, InferenceTaskWrapper):
            task.task.done()
        else:
            self.models[task.model].handle(
                InferenceTaskWrapper(task, self.queue.put))

    def run(self, queue, process_func):
        while True:
            try:
                request = queue.get()
                task = yield request
                # task: Union[InferenceTask, InferenceTaskWrapper] = yield request
                yield self.env.timeout(self.spec.route_latency.draw())
            except simpy.Interrupt:
                _LOGGER.debug(*self.log("interrupted"))
                request.cancel()
                if request.triggered and request._value is not None:
                    # put the task back to the store not to lose this request
                    task = request._value
                    queue.items.insert(0, task)
                    _LOGGER.debug(
                        *self.log("task (%s) is put back", hex(id(task))))
                break
            if task is None:
                # graceful stop
                _LOGGER.debug(*self.log("graceful stop"))
                break
            process_func(task)
        # drain
        for task in queue.items:
            yield self.env.timeout(self.spec.route_latency.draw())
            process_func(task)

    def deploy_model(self, spec: InferenceModelSpec):
        if spec.name in self.models:
            raise RuntimeError(f"{spec.name} was already deployed")
        self.models[spec.name] = InferenceModel(spec, self)

    def request_worker(self, resources: Resources) -> Tuple[Optional[Worker], bool]:
        """Returns (worker, is_new_pod_added)
        """
        _LOGGER.debug(*self.log("requesting %s", resources))
        if not resources <= self.spec.worker_spec:
            raise ValueError("requested resources exceed the worker spec")

        for worker in self.workers:
            avail = worker.available_resources()
            _LOGGER.debug(
                *self.log("worker (%s). avail=%s", hex(id(worker)), avail))
            if resources <= avail:
                _LOGGER.debug(*self.log("worker (%s) was provided for %s",
                                        hex(id(worker)), resources))
                return worker, False
        # trying to add a new worker if possible
        if (self.max_workers is not None and
                len(self.workers) >= self.max_workers):
            _LOGGER.debug(*self.log(
                "no available worker, but "
                "num_current_workers (%d) >= max_num_workers (%d)",
                len(self.workers), self.max_workers))
            return None, False
        _LOGGER.debug(
            *self.log("no available worker. try to add a new worker"))
        if self.add_worker():
            # successfully add a new worker. This should be the last worker.
            worker = self.workers[-1]
            assert resources <= worker.available_resources()
            _LOGGER.debug(*self.log("worker (%s) was provided for %s",
                                    hex(id(worker)), resources))
            return worker, True
        _LOGGER.debug(*self.log("failed to add a new worker"))
        return None, False

    def add_worker(self) -> bool:
        # quota check is done by the controller
        pod = self.controller.create_pod(self.spec.worker_spec, self.name)
        if pod is None:
            return False
        self.workers.append(Worker(pod))
        return True

    def remove_worker(self, worker: Worker = None) -> bool:
        if len(self.workers) <= self.spec.min_workers:
            _LOGGER.info(*self.log("no worker to remove. min_num_worker=%d",
                                   self.spec.min_workers))
            return False

        if worker is None:
            # pick an arbitrary worker.
            if len(self.workers) == 0:
                _LOGGER.warn(*self.log("no worker to remove exists"))
                return False
            worker = self.workers[-1]
        worker.stop()
        self.controller.remove_pod(worker.pod)
        self.workers.remove(worker)
        return True

    def stop(self):
        # graceful stop
        self.proc.interrupt()
        _LOGGER.debug(*self.log("interrupting proc"))
        yield self.proc
        _LOGGER.debug(*self.log("finish wait for the proc"))
        for model in self.models.values():
            yield self.env.process(model.stop())
        _LOGGER.debug(*self.log("finish wait for model stop"))
        if self.monitor_proc is not None:
            self.monitor_proc.interrupt()
            _LOGGER.debug(*self.log("interrupting monitor proc"))
            yield self.monitor_proc
            _LOGGER.debug(*self.log("finish wait for the monitor proc"))
