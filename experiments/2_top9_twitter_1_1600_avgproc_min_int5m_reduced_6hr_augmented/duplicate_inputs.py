import argparse
import json
from pathlib import Path

def duplicate_inputs(input_path, factor, output_path):
    out_lines = []

    for line in Path(input_path).read_text().splitlines():
        data = json.loads(line)
        count_dict = data["counts"]
        num_jobs = len(count_dict)
        new_count_dict = {
            f"serve-cluster{i}": count_dict[f"serve-cluster{i % num_jobs}"]
            for i in range(num_jobs * factor)
        }
        out_lines.append(json.dumps({
            "ts": data["ts"],
            "counts": new_count_dict}))

    Path(output_path).write_text("\n".join(out_lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--output_path", type=str, default="out.json")
    args = parser.parse_args()

    duplicate_inputs(args.input_path, args.factor, args.output_path)
