import argparse
import json
from pathlib import Path

import numpy as np


def split(config, count):
    urls = {}
    sum = 0
    for url, fraction in config.items():
        n = int(fraction * count)
        urls[url] = n
        sum += n
    # distribute remained requests
    indices = np.random.choice(len(config), count - sum)
    keys = list(urls.keys())
    for idx in indices:
        urls[keys[idx]] += 1
    return urls


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="count input file path")
    parser.add_argument("--config", required=True, help="config file path")
    parser.add_argument("--out", help="output file path")
    parser.add_argument("--reassign", action="store_true",
                        help="reassign input file with the given config")

    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines = f.readlines()

    with open(args.config, "r") as f:
        config = json.load(f)
    # check config is valid
    if abs(sum(list(config.values())) - 1.0) > 0.00001:
        raise ValueError("The sum of fractions (%f) != 1.0" %
                         sum(list(config.values())))

    if args.out:
        output_path = args.out
    else:
        output_path = Path(args.input).with_name("parsed.json")

    if args.reassign:
        print("Reassigning")
        new_entries = []
        for i, line in enumerate(lines):
            entry = json.loads(line)
            if len(config) != len(entry["urls"]):
                raise ValueError("Invalid config file")
            if i == 0:
                for old, new in zip(entry["urls"].keys(), config.keys()):
                    print(f"{old} -> {new}")
            new_entries.append({
                "ts": entry["ts"],
                "urls": {
                    url: count for url, count
                    in zip(config.keys(), entry["urls"].values())
                }})
        with open(output_path, "w") as f:
            for entry in new_entries:
                f.write(json.dumps(entry))
                f.write("\n")
    else:
        max_ts = 0
        total_counts = 0
        with open(output_path, "w") as f:
            for line in lines:
                ts, count = line.split(" ")
                count = int(count.strip())
                total_counts += count
                urls = split(config, count)
                assert sum(list(urls.values())) == count
                f.write(json.dumps({"ts": int(ts), "urls": urls}))
                f.write("\n")
                max_ts = max(max_ts, int(ts))

        print("Done")
        print("total counts: {}".format(total_counts))
        print("max ts: {}".format(max_ts))
