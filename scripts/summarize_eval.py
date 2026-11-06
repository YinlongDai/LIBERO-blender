"""Table of eval runs from scripts/evaluate.py: success per task and overall, with binomial standard errors.

    python scripts/summarize_eval.py results/eval/run_a results/eval/run_b --out results/eval/summary.md
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = {Path(run).name: [json.loads(p.read_text()) for p in sorted(Path(run).glob("*/task*.json"))]
            for run in args.runs}

    lines = ["# Eval summary", "", "| run | policy | images | success |", "|---|---|---|---|"]
    for name, results in runs.items():
        first = results[0]
        images = first["render_config"]["name"] if first["render_config"] else "mujoco"
        if first["lighting"]:
            images += f" + {first['lighting']['name']} v{first['lighting']['version']}"
        successes = [s for result in results for s in result["successes"]]
        rate = float(np.mean(successes))
        stderr = float(np.sqrt(rate * (1 - rate) / len(successes)))
        lines.append(f"| {name} | {first['policy']} | {images} | {rate:.3f} ± {stderr:.3f} ({len(successes)}) |")

    task_ids = sorted({result["task_id"] for results in runs.values() for result in results})
    lines += ["", "Per task:", "", "| task | " + " | ".join(runs) + " |", "|---|" + "---|" * len(runs)]
    for task_id in task_ids:
        cells = [next((f"{r['success_rate']:.2f}" for r in results if r["task_id"] == task_id), "-")
                 for results in runs.values()]
        lines.append(f"| {task_id} | " + " | ".join(cells) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
