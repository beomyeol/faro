import asyncio
import aiohttp
import time
import argparse
import numpy
import ray
from ray import serve
import multiprocessing as mp


async def get_async(url, payload, session, repeat, latencies):
    for _ in range(repeat):
        start_ts = time.time()
        async with session.get(url, data=payload) as response:
            if response.status != 200:
                print("Error status:", response.status)
                break
            await response.json()
            latencies.append((start_ts, time.time() - start_ts))


async def main(args):
    latencies = []
    url = "http://" + args.url
    async with aiohttp.ClientSession() as session:
        async with session.get(args.image_url) as response:
            if response.status != 200:
                print(f"Error in fetching image: status={response.status}")
                return
            payload = await response.read()
    print(f"image size: {len(payload)}")
    async with aiohttp.ClientSession() as session:
        start_ts = time.time()
        await asyncio.gather(*[get_async(url, payload, session, args.repeat, latencies) for _ in range(args.concurrent)])
        elapsed_time = time.time() - start_ts

    if args.print_raw:
        for ts, latency in latencies:
            print(f"{ts - start_ts} {latency}")
    else:
        _, latencies = zip(*latencies)
        print(f"average latency: {numpy.mean(latencies)}")
        print(f"min latency: {numpy.min(latencies)}")
        print(f"median latency: {numpy.percentile(latencies, 50)}")
        print(f"90 percentile: {numpy.percentile(latencies, 90)}")
        print(f"max latency: {numpy.max(latencies)}")
        print(f"total elapsed time: {elapsed_time}")


class ScaleOutManager(mp.Process):

    def __init__(self, stop_flag, period_s, ray_head_address, deployment_name, max_scale_out):
        super().__init__()
        self.period_s = period_s
        self.ray_head_address = ray_head_address
        self.deployment_name = deployment_name
        self.max_scale_out = max_scale_out
        self.stop_flag = stop_flag

    def run(self):
        ray.init(address=self.ray_head_address)
        deployment = serve.get_deployment(self.deployment_name)

        self.stop_flag.value = False

        for num_replica in range(2, self.max_scale_out + 1):
            time.sleep(self.period_s)
            if self.stop_flag.value:
                break
            deployment.options(num_replicas=num_replica).deploy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="url")
    parser.add_argument("--concurrent", type=int, default=1,
                        help="num concurrent requests")
    parser.add_argument("--repeat", type=int, default=50,
                        help="number of request repeats")
    parser.add_argument("--image-url", required=True, help="image url")
    parser.add_argument("--scale-out-period", type=int,
                        help="scale out period")
    parser.add_argument("--max-scale-out", type=int, help="maximum replicas")
    parser.add_argument("--ray-head-address", help="ray head adddress")
    parser.add_argument("--serve-name", help="deployment name")
    parser.add_argument("--print-raw", action="store_true",
                        help="print raw latency data")
    args = parser.parse_args()

    manager = None
    if args.scale_out_period:
        assert args.scale_out_period is not None
        assert args.max_scale_out is not None
        assert args.ray_head_address is not None
        assert args.serve_name is not None
        stop_flag = mp.Value("b", True)
        manager = ScaleOutManager(stop_flag, args.scale_out_period,
                                  args.ray_head_address,
                                  args.serve_name, args.max_scale_out)
        manager.start()

        while not stop_flag.value:
            time.sleep(0.1)

    asyncio.run(main(args))

    if manager is not None:
        stop_flag.value = True
        manager.join()
