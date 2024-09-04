from typing import Dict, List
from collections import defaultdict
from math import ceil
import os

import numpy as np

# autoscaling look back window size to aggregate
LOOK_BACK_PERIOD_S: float = float(os.environ.get("LOOK_BACK_PERIOD_S", 30.0))

# interval to send metrics from replicas
INTERVAL_S: float = float(os.environ.get("INTERVAL_S", 10.0))

QUEUE_LEN_KEY = "queue_len"
LATENCY_KEY = "avg_latency"
INPUT_RATE_KEY = "input_rate"
HEAD_QUEUE_LEN_KEY = "head_queue_len"
DROP_RATE_KEY = "drop_rate"

class TimeMetricStore:

    def __init__(self):
        self.data = defaultdict(list)

    def record(self, name: str, timestamp: float):
        self.data[name].append(timestamp)

    def hist(self, bin_size_s: float, current_ts: float, clear: bool = True) -> Dict[str, List[int]]:
        hist_dict = {}

        for name, tss in self.data.items():
            if len(tss) == 0:
                continue

            num_bins = max(ceil((current_ts - tss[0]) / bin_size_s), 1)
            bins = np.arange(0, bin_size_s * num_bins + 1, bin_size_s)
            tss = [abs(ts - current_ts) for ts in tss]
            counts, bin_edges = np.histogram(tss, bins=bins)
            hist_dict[name] = (np.flip(counts).tolist(), tss[0])

        if clear:
            self.data.clear()

        return hist_dict
