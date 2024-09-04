import argparse
from collections import defaultdict
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input json file")
    parser.add_argument("-o", "--output", default="sim_input.json",
                        help="output json file for simluation")

    args = parser.parse_args()

    cluster_map = {}

    input_path = Path(args.input)
    output_lines = []
    for line in input_path.read_text().splitlines():
        obj = json.loads(line)
        count_dict = defaultdict(dict)
        for url, count in obj["urls"].items():
            info = cluster_map.get(url, None)
            if info is None:
                job = url[url.rfind("/")+1:]
                cluster_url = url[:url.rfind("/")]
                cluster = "cluster" + str(len(cluster_map) + 1)
                cluster_map[url] = (cluster, job)
            else:
                cluster, job = info
            count_dict[cluster][job] = count
        output_lines.append(json.dumps({
            "ts": obj["ts"],
            "counts": count_dict,
        }))

    output_path = Path(args.output)
    print("Outputting to %s..." % output_path)
    output_path.write_text(
        "\n".join(output_lines)
    )
