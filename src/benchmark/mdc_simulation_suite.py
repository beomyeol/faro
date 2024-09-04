import argparse
from copy import copy
from concurrent.futures import ProcessPoolExecutor

from benchmark.mdc_simulation import main as mdc_simluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-rates", required=True)
    # parser.add_argument("--input-rate", type=int)
    parser.add_argument("--resource-limit", type=int, default=48)
    parser.add_argument("--mdc-percentile", type=int)
    parser.add_argument("--max-requests", type=int, default=5000)
    parser.add_argument("--num-replicas", type=int)
    args = parser.parse_args()

    executor = ProcessPoolExecutor()

    input_rates = [int(v) for v in args.input_rates.split(",")]
    futures = []
    for input_rate in input_rates:
        sim_args = copy(args)
        sim_args.input_rate = input_rate

        futures.append(executor.submit(mdc_simluation, sim_args))

    for future in futures:
        outputs = future.result()
        print("\t".join([
            f'{outputs["input_rate"]}',
            f'{outputs["mdc"]}',
            f'{outputs["num_replicas"]}',
            f'{outputs["latency"]}',
            f'{outputs["route_latency"]}',
            f'{outputs["num_requests"]}',
            f'{outputs["avg"]:.2f}',
            f'{outputs["max"]:.2f}',
            f'{outputs["SLO violation"]:.2f}%',
            f'{outputs["processing_time"]:.2f}',
        ]))
