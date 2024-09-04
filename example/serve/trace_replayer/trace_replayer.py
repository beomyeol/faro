from asyncio.log import logger
from gevent import monkey
monkey.patch_all()  # nopep8
import gevent
import gevent.queue
import argparse
import time
import json
import requests
import numpy as np
from typing import List
import logging
import sys

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_HANDLER = logging.StreamHandler(sys.stderr)
_HANDLER.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'))
_LOGGER.addHandler(_HANDLER)


class Producer:

    def __init__(self, queue: gevent.queue.Channel, input_path):
        self.queue = queue
        self.input_path = input_path
        self.greenlet: gevent.Greenlet = None

    def produce_per_url(self, url, count):
        interval = 0.5 / count
        for _ in range(count):
            self.queue.put(url)
            gevent.sleep(interval)

    def run(self):
        try:
            _LOGGER.info("[producer] run")
            with open(self.input_path, "r") as f:
                entries = [json.loads(line) for line in f]

            start_ts = time.time()

            for entry in entries:
                ts = entry["ts"]
                if ts > 0:
                    time_to_sleep = start_ts + ts - time.time()
                    if time_to_sleep < 0:
                        # _LOGGER.error("[producer] time is off")
                        # break
                        raise RuntimeError(
                            f"time is off: ts={ts}, sleep_time={time_to_sleep}")
                    gevent.sleep(time_to_sleep)

                glets = []
                for url, count in entry["urls"].items():
                    if count == 0:
                        continue
                    glets.append(gevent.spawn(self.produce_per_url, url, count))
                gevent.joinall(glets)

        except gevent.GreenletExit:
            _LOGGER.info("[producer] stopping")

    def start(self):
        self.greenlet = gevent.spawn(self.run)

    def stop(self):
        self.greenlet.kill()
        self.join()

    def join(self):
        self.greenlet.join()


class Stats:

    def __init__(self):
        self.start_tss = []
        self.latencies = []

    def log(self, start_ts, latency):
        self.start_tss.append(start_ts)
        self.latencies.append(latency)

    def avg(self):
        return np.mean(self.latencies)

    def count(self):
        return len(self.latencies)

    def write_latencies(self, path):
        with open(path, "w") as f:
            for latency in self.latencies:
                f.write(f"{latency}\n")

    def write_start_times(self, base_ts, path):
        with open(path, "w") as f:
            for start_ts in self.start_tss:
                f.write(f"{start_ts - base_ts}\n")


class Worker:

    def __init__(self, queue: gevent.queue.Channel, stats: Stats, image_bytes: bytes):
        self.queue = queue
        self.stats = stats
        self.greenlet: gevent.Greenlet = None
        self.image_bytes = image_bytes

    def start(self):
        self.greenlet = gevent.spawn(self.run)

    def stop(self):
        self.greenlet.kill()
        self.join()

    def join(self):
        self.greenlet.join()

    def run(self):
        try:
            session = requests.Session()
            while True:
                url = self.queue.get()
                start_pc = time.perf_counter()
                r = session.get(url, data=self.image_bytes)
                if r.status_code != 200:
                    raise RuntimeError(f"status code: {r.status_code}")
                self.stats.log(start_pc, time.perf_counter() - start_pc)
        except gevent.GreenletExit:
            pass


def main(args):
    with open(args.img, "rb") as f:
        image_bytes = f.read()

    queue = gevent.queue.Channel()
    stats = Stats()

    workers: List[Worker] = []
    _LOGGER.info("# workers: %d", args.num_workers)
    for _ in range(args.num_workers):
        worker = Worker(queue, stats, image_bytes)
        worker.start()
        workers.append(worker)

    producer = Producer(queue, args.input)
    start_ts = time.perf_counter()
    producer.start()
    producer.join()

    for worker in workers:
        worker.stop()

    print(stats.count())
    print(f"{stats.avg() * 1e3} ms")

    if args.out:
        stats.write_latencies(args.out)

    if args.start_time_out:
        stats.write_start_times(start_ts, args.start_time_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file path")
    parser.add_argument("--num_workers", type=int,
                        default=10, help="# workers")
    parser.add_argument("--img", required=True, help="image file path")
    parser.add_argument("--out", help="latency output path")
    parser.add_argument("--start_time_out", help="start time output path")
    args = parser.parse_args()
    main(args)
