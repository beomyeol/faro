from collections import defaultdict
import itertools
import logging
import random
from typing import List, TYPE_CHECKING, Optional
from dataclasses import dataclass

import simpy

from simulation.distribution import Distribution
from simulation.env import global_env
from simulation.primitives import Resources, Job, Task
from simulation import metric

if TYPE_CHECKING:
    from simulation.cluster import Cluster

_LOGGER = logging.getLogger(__name__)


class InferenceTask(Task):

    def __init__(self, cluster_name, model) -> None:
        super().__init__(cluster_name)
        self.model = model

    def __repr__(self) -> str:
        return "InferenceTask({0}, {1})".format(
            self.cluster_name, self.model)


@dataclass
class InferenceModelSpec:
    cluster_name: str
    name: str
    num_replicas: int
    resources: Resources
    latency_dist: Distribution
    max_concurrent_queries: int
    max_queue_len: int


class TaskSchedulerWrapper:

    def __init__(self, task: Task, callback):
        self.task = task
        self.callback = callback
        self.replica_done_event = global_env().event()

    @property
    def start_ts(self):
        return self.task.start_ts

    def arrive(self):
        self.task.arrive()

    def start(self):
        self.task.start()

    def done(self, succeeded: bool = True):
        _LOGGER.debug("done event (%s) is triggered for task (%s)",
                      hex(id(self.replica_done_event)), hex(id(self)))
        self.replica_done_event.succeed()
        self.callback(self.replica_done_event)
        self.task.done(succeeded)

    def stats(self):
        return self.task.stats()


class InferenceModel:

    def __init__(self, spec: InferenceModelSpec, cluster) -> None:
        self.env = global_env()
        self.spec = spec
        self.cluster: Cluster = cluster
        self.replicas: List[InferenceModelReplica] = []

        _LOGGER.info(*self.log("%s", self.spec))

        self.queue = simpy.Store(self.env)
        self.scheduler = self.env.process(self.schedule())
        self.config_update_event = self.env.event()

        self.add_replica(self.spec.num_replicas)
        self.replica_iterator = itertools.cycle(self.replicas)
        self.is_stopped = False
        self._drop_rate = 0.0

    @property
    def name(self):
        return self.spec.name

    @property
    def num_replicas(self):
        return len(self.replicas)

    def get_queue_len(self):
        return len(self.queue.items)

    def drop_rate(self):
        return self._drop_rate

    def set_drop_rate(self, drop_rate):
        _LOGGER.info(*self.log("set drop rate: %.4f", drop_rate))
        self._drop_rate = drop_rate

    def queue_len_metrics(self) -> List[int]:
        return [replica.get_queue_len() for replica in self.replicas]

    def log(self, msg, *args):
        new_msg = "[%.3f] (%s:%s) " + msg
        new_args = (self.env.now, self.cluster.name, self.name, *args)
        return (new_msg, *new_args)

    def _reset_replica_iterator(self):
        _LOGGER.info(*self.log("#replicas: %d", len(self.replicas)))
        replicas = list(self.replicas)
        random.shuffle(replicas)
        self.replica_iterator = itertools.cycle(replicas)
        if self.config_update_event.triggered:
            self.config_update_event.succeed()

    def _add_replica(self, num: int = 1):
        _LOGGER.debug(*self.log("adding %d replicas. current #replicas=%d",
                                num, len(self.replicas)))

        num_new_pod_added = 0
        replicas = []
        for _ in range(num):
            worker, is_new_pod_added = self.cluster.request_worker(
                self.spec.resources)
            if worker is None:
                _LOGGER.warn(
                    *self.log("failed to add a replicas. no available worker"))
                break
            if is_new_pod_added:
                num_new_pod_added += 1
            replica = InferenceModelReplica(self)
            replica.set_worker(worker)
            _LOGGER.debug(*self.log(
                "new replica (%s) is added to worker (%s, avail=%s). "
                "#replicas=%d",
                hex(id(replica)), hex(id(worker)),
                worker.available_resources(), len(self.replicas)))
            replicas.append(replica)

        return replicas, num_new_pod_added

    def add_replica_generator(self, num: int = 1,
                              max_trials: Optional[int] = 1,
                              backoff_s: float = 1.0):

        def _add_replicas(replicas, num_new_pod_added):
            if num_new_pod_added > 0:
                _LOGGER.info(
                    *self.log("Cold start %d pods", num_new_pod_added))
                hot_replicas = replicas[:-num_new_pod_added]
                cold_replicas = replicas[-num_new_pod_added:]
            else:
                hot_replicas = replicas
                cold_replicas = []
            # hot start
            if len(hot_replicas) > 0:
                yield self.env.timeout(self.cluster.controller.params["hot_start_overhead_s"])
                if self.is_stopped:
                    return
                _LOGGER.info(
                    *self.log("Starting %d hot replicas", len(hot_replicas)))
                for replica in hot_replicas:
                    replica.start()
                self.replicas.extend(hot_replicas)
                self._reset_replica_iterator()
            # cold start
            if len(cold_replicas) > 0:
                timeout = self.cluster.controller.params["cold_start_overhead_s"]
                if len(hot_replicas) > 0:
                    timeout -= self.cluster.controller.params["hot_start_overhead_s"]
                yield self.env.timeout(timeout)
                if self.is_stopped:
                    return
                _LOGGER.info(
                    *self.log("Starting %d cold replicas", len(cold_replicas)))
                for replica in cold_replicas:
                    replica.start()
                self.replicas.extend(cold_replicas)
                self._reset_replica_iterator()

        procs = []
        trials = 0

        while not self.is_stopped:
            trials += 1
            metric.try_scale_out(self.cluster.name, num)
            replicas, num_new_pod_added = self._add_replica(num)
            metric.scale_out_done(self.cluster.name, len(replicas))
            procs.append(self.env.process(
                _add_replicas(replicas, num_new_pod_added)))

            left_num = num - len(replicas)
            if left_num > 0 and (max_trials is None or trials < max_trials):
                if max_trials is not None:
                    _LOGGER.warn(*self.log(
                        "failed to add %d/%d replicas. retry (%d/%d) after %.2f s",
                        left_num, num, trials, max_trials, backoff_s))
                yield self.env.timeout(backoff_s)
                num = left_num
            else:
                break

        yield simpy.AllOf(self.env, procs)

    def add_replica(self, num: int = 1) -> bool:
        replicas, _ = self._add_replica(num)
        self.replicas.extend(replicas)
        for replica in replicas:
            replica.start()
        self._reset_replica_iterator()
        return True

    def remove_replica(self, num: int = 1):
        # TODO: inteligent pick
        for _ in range(num):
            replica = self.replicas.pop()
            replica.unset_worker()
            replica.stop()
            _LOGGER.debug(*self.log("replica (%s) is removed. #replicas=%d",
                                    hex(id(replica)), len(self.replicas)))
            metric.scale_in_done(self.cluster.name)
        self._reset_replica_iterator()

    def remove_replica_generator(self, num: int = 1, delay: int = 0):
        yield self.env.timeout(delay)
        if self.is_stopped:
            return
        self.remove_replica(num)
        timeout = self.cluster.controller.params["hot_start_overhead_s"]
        yield self.env.timeout(timeout)

    def handle(self, task: InferenceTask):
        if random.random() < self._drop_rate:
            # drop based on the drop rate
            task.done(succeeded=False)
            return

        if (self.spec.max_queue_len > 0 and
                self.get_queue_len() > self.spec.max_queue_len):
            # drop this to avoid OOM. This task also is unlikely to complete
            # within the SLO
            task.done(succeeded=False)
        else:
            self.queue.put(task)

    def schedule(self):
        inflight_task_events = defaultdict(list)

        def _assign():
            for _ in self.replicas:
                replica = next(self.replica_iterator)
                num_ongoing_tasks = len(inflight_task_events[replica])
                if num_ongoing_tasks >= self.spec.max_concurrent_queries:
                    _LOGGER.debug(*self.log(
                        "replica (%s) cannot be used. #ongoing: %d, limit: %d",
                        hex(id(replica)), num_ongoing_tasks,
                        self.spec.max_concurrent_queries))
                    continue
                return replica
            return None

        _LOGGER.info(*self.log("scheduler gets started"))
        while not self.is_stopped or len(self.queue.items) > 0:
            try:
                request = self.queue.get()
                task = yield request
            except simpy.Interrupt:
                _LOGGER.debug(*self.log("scheduler interrupted"))
                request.cancel()
                if request.triggered and request._value is not None:
                    raise NotImplementedError("pending task is not handled.")
                break

            target_replica = _assign()
            while target_replica is None:
                # failed to assign need to wait for until either any ongoing
                # tasks are done or new replicas are added
                target_events = [self.config_update_event]
                target_events.extend(itertools.chain(
                    *inflight_task_events.values()))
                _LOGGER.debug(
                    *self.log("waiting for fininshed tasks or new replicas: %s",
                              [hex(id(event)) for event in target_events]))
                yield simpy.AnyOf(self.env, target_events)
                _LOGGER.debug(*self.log("waiting is done"))
                if self.config_update_event.triggered:
                    self.config_update_event._value = simpy.events.PENDING
                    self.config_update_event.callbacks = []
                target_replica = _assign()

            target_task_events = inflight_task_events[target_replica]
            task = TaskSchedulerWrapper(task, target_task_events.remove)
            target_task_events.append(task.replica_done_event)
            target_replica.handle(task)

        _LOGGER.info(*self.log("stopping scheduler"))
        assert len(self.queue.items) == 0

    def stop(self):
        self.is_stopped = True
        # put None task to stop replicas
        # Since store is processed in the FIFO manner, tasks queued before
        # should be processed correctly
        _LOGGER.debug(*self.log("stopping %d replicas", len(self.replicas)))
        for replica in self.replicas:
            replica.graceful_stop()

        # wait for the completion of all replicas
        procs = [replica.proc for replica in self.replicas]
        _LOGGER.debug(
            *self.log("wait for the completion of %d replicas", len(self.replicas)))
        while True:
            yield simpy.AllOf(self.env, procs)
            _LOGGER.debug(*self.log("done. check changes"))
            # double check whether there is any change in replicas
            new_procs = [replica.proc for replica in self.replicas]
            if procs == new_procs:
                break
            procs = new_procs

        self.scheduler.interrupt()
        _LOGGER.debug(*self.log("waiting for the scheduler"))
        yield self.scheduler
        _LOGGER.debug(*self.log("done"))


class InferenceModelReplica(Job):

    def __init__(self, model: InferenceModel):
        super().__init__(model.spec.resources)
        self._model = model
        self.proc: simpy.Process = None
        self.queue = simpy.Store(self.env)

    @property
    def env(self):
        return self._model.env

    def get_queue_len(self):
        return len(self.queue.items)

    def handle(self, task: InferenceTask):
        return self.queue.put(task)

    def get_latency(self):
        return self._model.spec.latency_dist.draw()

    def log(self, msg, *args):
        new_msg = "[%.3f] (%s, %s) replica (%s) " + msg
        new_args = (self.env.now, self._model.cluster.name,
                    self._model.name, hex(id(self)), *args)
        return (new_msg, *new_args)

    def start(self):
        self.proc = self._model.env.process(self.run())

    def stop(self):
        if self.proc is not None:
            self.proc.interrupt()

    def graceful_stop(self):
        return self.queue.put(None)

    def run(self):
        _LOGGER.debug("[%f], replica (%s, %s) gets started",
                      self.env.now, self._model.cluster.name, self._model.name)
        while True:
            try:
                request = self.queue.get()
                # task: InferenceTask = yield request
                task = yield request
            except simpy.Interrupt:
                _LOGGER.debug(*self.log("interrupted"))
                request.cancel()
                if request.triggered and request._value is not None:
                    # hack. put the task back to the store.
                    task = request._value
                    self.queue.items.insert(0, task)
                    _LOGGER.debug(
                        *self.log("task (%s) is put back", hex(id(task))))
                break
            if task is None:
                # graceful stop
                _LOGGER.debug(*self.log("graceful stop"))
                break
            task.start()
            _LOGGER.debug(*self.log("task (%s) starts", hex(id(task))))
            try:
                latency = self.get_latency()
                yield self.env.timeout(latency)
            except simpy.Interrupt:
                _LOGGER.debug(*self.log(
                    "interrupted while running task (%s). fininsh it",
                    hex(id(task))))
                remain_time = (task.start_ts + latency) - self.env.now
                yield self.env.timeout(remain_time)
                task.done()
                _LOGGER.debug(*self.log("task (%s) done. %s",
                                        hex(id(task)), task.stats()))
                break
            task.done()
            _LOGGER.debug(*self.log("task (%s) done. %s",
                                    hex(id(task)), task.stats()))

        _LOGGER.debug(*self.log("interruted processing tasks in the queue"))
        # processing tasks in the queue
        # no item will be queued since this process is already stopped.
        # see InferenceModel.remove_replica()
        for task in self.queue.items:
            task.start()
            latency = self.get_latency()
            yield self.env.timeout(latency)
            task.done()
            _LOGGER.debug(*self.log("task (%s) done. %s",
                                    hex(id(task)), task.stats()))
        _LOGGER.debug(*self.log("done"))
