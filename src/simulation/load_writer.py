import argparse
from io import StringIO, TextIOWrapper
import logging
from typing import List

import numpy as np
import simpy

_LOGGER = logging.getLogger(__name__)


class Client:

    def __init__(self, env: simpy.Environment, name: str, interval: float, f: TextIOWrapper) -> None:
        self.env = env
        self.name = name
        self.interval = interval
        self.f = f

        self.num_queries = 0

    def run(self):
        while True:
            yield self.env.timeout(np.random.exponential(self.interval))
            self.f.write(f"{self.name},{self.env.now}\n")
            self.num_queries += 1


def main(args):
    if args.qps:
        intervals = [1 / float(qps) for qps in args.qps.split(",")]
    else:
        intervals = [float(interval) for interval in args.interval.split(",")]

    env = simpy.Environment()

    _LOGGER.info("# clients: %d", len(intervals))
    _LOGGER.info("intervals: %s", str(intervals))
    _LOGGER.info("writing to %s...", args.out)
    _LOGGER.info("until: %.4f", args.until)

    with open(args.out, "w") as f:
        f.write("func,start_timestamp\n")
        clients: List[Client] = []
        for i, interval in enumerate(intervals):
            client = Client(env, f"f_{i}", interval, f)
            env.process(client.run())
            clients.append(client)

        env.run(until=args.until)
        num_queries_log = StringIO()
        num_queries_log.write("[num queries]")
        for client in clients:
            num_queries_log.write(f"\n  {client.name}: {client.num_queries}")
        _LOGGER.info(num_queries_log.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output.csv", help="output csv path")
    parser.add_argument("--until", required=True, type=float,
                        help="max time in second")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--qps", help="comma-separated qps per model")
    group.add_argument("--interval", help="comma-separated interval per model")
    main(parser.parse_args())
