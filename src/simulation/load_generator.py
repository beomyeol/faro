from __future__ import annotations

from abc import abstractmethod
import csv
import json
import logging
from typing import Tuple

import simpy
import numpy as np

from simulation.controller import Controller
from simulation.primitives import Task
from simulation.inference import InferenceTask

_LOGGER = logging.getLogger(__name__)


class LoadGenerator:

    def __init__(self, controller: Controller):
        self.controller = controller
        self.is_stopped = False
        self.proc: simpy.Process = None

    @property
    def env(self) -> simpy.Environment:
        return self.controller.env

    @abstractmethod
    def next(self) -> Tuple[float, Task]:
        pass

    def log(self, msg, *args):
        new_msg = "[%.3f] generator(%s) " + msg
        new_args = (self.env.now, hex(id(self)), *args)
        return (new_msg, *new_args)

    def run(self):
        def _run():
            task_done_events = []
            while not self.is_stopped:
                try:
                    interval, task = self.next()
                except StopIteration:
                    break
                try:
                    yield self.env.timeout(interval)
                except simpy.Interrupt:
                    _LOGGER.debug(*self.log("stopped"))
                    break
                self.controller.submit(task)
                task_done_events.append(task.done_event)
            yield simpy.AllOf(self.env, task_done_events)

        self.proc = self.env.process(_run())

    def stop(self):
        self.is_stopped = True
        self.proc.interrupt()

    def stop_at(self, until: int):
        def _stop(client):
            yield self.env.timeout(until)
            client.stop()
        self.env.process(_stop(self))


class FixedIntervalLoadGenerator(LoadGenerator):

    def __init__(self, controller: Controller, interval_s: float,
                 cluster_name: str, job_name: str,
                 max_num_tasks: int = None):
        super().__init__(controller)
        self.interval_s = interval_s
        self.cluster_name = cluster_name
        self.job_name = job_name
        self.max_num_tasks = max_num_tasks
        self.num_tasks = 0

    def next(self):
        if (self.max_num_tasks is not None and
                self.num_tasks >= self.max_num_tasks):
            raise StopIteration()
        task = InferenceTask(self.cluster_name, self.job_name)
        _LOGGER.debug(*self.log("%s [%s] is generated",
                      task, hex(id(task))))
        self.num_tasks += 1
        return self.interval_s, task


class PoissonLoadGenerator(LoadGenerator):

    def __init__(self, controller: Controller, interval_s: float,
                 cluster_name: str, job_name: str,
                 max_num_tasks: int = None):
        super().__init__(controller)
        self.interval_s = interval_s
        self.cluster_name = cluster_name
        self.job_name = job_name
        self.max_num_tasks = max_num_tasks
        self.num_tasks = 0

    def next(self):
        if (self.max_num_tasks is not None and
                self.num_tasks >= self.max_num_tasks):
            raise StopIteration()
        self.num_tasks += 1
        task = InferenceTask(self.cluster_name, self.job_name)
        interval = np.random.exponential(self.interval_s)
        _LOGGER.debug(*self.log("%s [%s] is generated. interval=%.3f",
                      task, hex(id(task)), interval))
        return interval, task


class CSVLoadGenerator(LoadGenerator):

    def __init__(self, controller: Controller,
                 path: str) -> None:
        super().__init__(controller)
        self.csv_file = open(path, newline="")
        self.reader = csv.DictReader(self.csv_file)
        self.prev_start_ts = 0

    def next(self):
        try:
            row = next(self.reader)
            start_ts = float(row["start_timestamp"])
            func_name = row["func"]
            interval = start_ts - self.prev_start_ts
            _LOGGER.debug(*self.log("func=%s, start_ts=%.4f, interval=%.4f",
                                    func_name, start_ts, interval))
            self.prev_start_ts = start_ts
            task = InferenceTask(func_name, func_name)
            return interval, task
        except StopIteration as e:
            # done
            self.csv_file.close()
            raise e

    def stop(self):
        self.csv_file.close()
        super().stop()


class JSONLoadGenerator:

    def __init__(self, controller: Controller, path: str,
                 overhead_s: float = 0.0, unit_time: int = 1,
                 poisson: bool = False, send_over_unit_time: bool = False,
                 instantly: bool = False, seed: int = 42):
        self.controller = controller
        self.is_stopped = False
        self.proc: simpy.Process = None
        self.path = path
        self.overhead_s = overhead_s
        self.unit_time = unit_time
        self.poisson = poisson
        self.send_over_unit_time = send_over_unit_time
        self.instantly = instantly
        self.rng = np.random.default_rng(seed)

        _LOGGER.info(
            "unit_time: %d, possion: %s, send_over: %s, instantly: %s, path: %s"
            ", seed: %d",
            unit_time, poisson, send_over_unit_time, instantly, path, seed)

    @property
    def env(self) -> simpy.Environment:
        return self.controller.env

    def log(self, msg, *args):
        new_msg = "[%.3f] generator(%s) " + msg
        new_args = (self.env.now, hex(id(self)), *args)
        return (new_msg, *new_args)

    def run(self):
        def _run_inner(cluster: str, job: str, count: int):
            if self.instantly:
                intervals = np.repeat(0, count)
            elif self.poisson:
                intervals = self.rng.exponential(
                    self.unit_time / count, count)
            else:
                interval = self.unit_time / count
                if not self.send_over_unit_time:
                    interval /= 2  # send over the first half unit time
                _LOGGER.debug(*self.log("interval for %s[%s] = %.2f",
                                        cluster, job, interval))
                intervals = np.repeat(interval, count)
            try:
                task_done_events = []
                for i in range(count):
                    task = InferenceTask(cluster, job)
                    _LOGGER.debug(*self.log("%s [%s] is generated",
                                            task, hex(id(task))))
                    self.controller.submit(task)
                    task_done_events.append(task.done_event)
                    yield self.env.timeout(self.overhead_s + intervals[i])
                yield simpy.AllOf(self.env, task_done_events)
            except simpy.Interrupt:
                _LOGGER.debug(*self.log("stopped"))

        def _run():
            _LOGGER.info(*self.log("opening %s...", self.path))
            with open(self.path, "r") as f:
                lines = f.readlines()

            all_procs = []
            for line in lines:
                if self.is_stopped:
                    break
                obj = json.loads(line)
                count_dict = obj["counts"]
                procs = []
                for cluster, jobs in count_dict.items():
                    for job, count in jobs.items():
                        if count == 0:
                            continue
                        procs.append(
                            self.env.process(_run_inner(cluster, job, count)))
                all_procs.extend(procs)
                yield self.env.timeout(self.unit_time)
            # wait for the proceeses are complete
            yield simpy.AllOf(self.env, all_procs)

        self.proc = self.env.process(_run())

    def stop(self):
        self.is_stopped = True
        self.proc.interrupt()

    def stop_at(self, until: int):
        def _stop(client):
            yield self.env.timeout(until)
            client.stop()
        self.env.process(_stop(self))


class ContinuousLoadGenerator:

    def __init__(self, controller: Controller, cluster_name: str,
                 job_name: str, num_tasks: int, overhead_s: float = 0.0):
        self.controller = controller
        self.is_stopped = False
        self.cluster_name = cluster_name
        self.job_name = job_name
        self.num_tasks = num_tasks
        self.proc: simpy.Process = None
        self.overhead_s = overhead_s

    @property
    def env(self):
        return self.controller.env

    def run(self):
        def _run():
            for _ in range(self.num_tasks):
                if self.is_stopped:
                    break
                try:
                    task = InferenceTask(self.cluster_name, self.job_name)
                    self.controller.submit(task)
                    yield self.env.timeout(self.overhead_s)
                    yield task.done_event
                except simpy.Interrupt:
                    _LOGGER.debug(*self.log("stopped"))
                    break
        self.proc = self.env.process(_run())

    def stop(self):
        self.is_stopped = True
        self.proc.interrupt()

    def stop_at(self, until: int):
        def _stop(client):
            yield self.env.timeout(until)
            client.stop()
        self.env.process(_stop(self))


_CLASS_MAPPING = {
    "poisson": PoissonLoadGenerator,
    "fixed": FixedIntervalLoadGenerator,
    "csv": CSVLoadGenerator,
    "json": JSONLoadGenerator,
    "continuous": ContinuousLoadGenerator,
}


def create_loadgen(type: str, *args, **kwargs) -> LoadGenerator:
    return _CLASS_MAPPING[type](*args, **kwargs)
