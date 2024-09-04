from abc import ABC, abstractmethod
import logging
from math import ceil
from typing import List, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)


class Predictor(ABC):

    @abstractmethod
    def predict(self, input_metrics, current_ts: float) -> Optional[List[int]]:
        pass


class BasePredictor(Predictor):

    def __init__(self, unit_time_s):
        self.unit_time_s = unit_time_s

        # TODO: This cannot generate history before the first call
        # if there is no request. can we solve this?
        self.prev_ts = None
        self.history = np.array([], dtype=np.float32)

    @property
    @abstractmethod
    def context_len(self) -> int:
        pass

    def update_history(self, input_metrics, current_ts):
        if self.prev_ts is None:
            self.prev_ts = current_ts
            if input_metrics is not None:
                counts, _ = input_metrics
                self.history = np.array(counts, dtype=np.float32)
        elif current_ts - self.prev_ts < 0.1:
            # another call to change quantile.
            # do not update history
            pass
        else:
            self.history = np.pad(
                self.history,
                (0, int(round(current_ts - self.prev_ts, 8) / self.unit_time_s)),
                constant_values=0)
            self.prev_ts = current_ts
            if input_metrics is not None:
                new_counts, max_ts = input_metrics
                begin = min(ceil(max_ts / self.unit_time_s), len(new_counts))
                self.history[-begin:] += new_counts

        if len(self.history) > self.context_len:
            self.history = self.history[-self.context_len:]
        return len(self.history) >= self.context_len


class OraclePredictor(Predictor):

    def __init__(self, input_counts: List[int], window_size_s: int,
                 head_name: str, deployment_name: str, unit_time_s: int = 1):
        self.input_counts = input_counts
        self.head_name = head_name
        self.first_req_ts = None
        assert window_size_s % unit_time_s == 0
        self.window_size = window_size_s // unit_time_s
        self.deployment_name = deployment_name
        self.unit_time_s = unit_time_s

        self.count_sum = 0
        # set non-zero idx for the case where no input exists at the beginning
        # otherwise, the current step calculation is from the first request,
        # not from the beginning of the inputs
        self.non_zero_idx = 0
        for count in self.input_counts:
            if count > 0:
                break
            self.non_zero_idx += 1
        self.input_cumsum = np.cumsum(self.input_counts)
        self.prev_ts = None

    def predict(self, input_metrics, current_ts: float) -> Optional[List[int]]:
        # _LOGGER.info("[%.3f] (%s:%s) metrics: %s", current_ts, self.head_name,
        #              self.deployment_name, input_metrics)
        if self.prev_ts is not None and current_ts - self.prev_ts < 0.1:
            # another call to change quantile.
            # do not internals
            pass
        else:
            self.prev_ts = current_ts
            if input_metrics is not None:
                counts, max_ts = input_metrics
                self.count_sum += np.sum(counts)
                if self.first_req_ts is None:
                    self.first_req_ts = current_ts - max_ts

        if self.first_req_ts is None:
            # does not have any information
            return None

        # TODO: ceil? or round?
        current_step = ceil(current_ts - self.first_req_ts) + self.non_zero_idx
        current_step //= self.unit_time_s  # TODO: ceil?
        if current_step >= len(self.input_counts):
            _LOGGER.info(
                "[%.3f] (%s:%s) Current step %d is out of range (max=%d)",
                current_ts, self.head_name, self.deployment_name,
                current_step, len(self.input_counts))
            return None
        _LOGGER.info(
            "[%.3f] (%s:%s) step: %d, counts: %d, expected: %d",
            current_ts, self.head_name, self.deployment_name,
            current_step, self.count_sum,
            self.input_cumsum[current_step - 1] if current_step > 0 else 0)
        return self.input_counts[current_step:current_step+self.window_size]
