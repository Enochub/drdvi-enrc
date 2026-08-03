"""Run selected methods on selected datasets and aggregate comparable scores."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

DATASETS = ("stickfigures", "cmnist", "nr_objects", "gtsrb")
METHODS = ("enrc", "drdvi", "frozen", "joint", "full_joint")
ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--out-root", default=str(ROOT / "results" / "comparison"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Override every training stage with this epoch count.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    failures, rows = [], []
    for dataset in args.datasets:
        config = ROOT / "configs" / f"{dataset}.yaml"
        jobs = []
        baselines = [method for method in args.methods if method != "joint"]
        if baselines:
            jobs.append(("baselines", ROOT / "experiments" / "frozen_drdvi_enrc.py", ["--methods", *baselines]))
        if "joint" in args.methods:
            jobs.append(("joint", ROOT / "experiments" / "joint_drdvi_enrc.py", []))
        if "full_joint" in args.methods:
            jobs.append(("full_joint", ROOT / "experiments" / "joint_full_drdvi_enrc.py", []))
        for job_name, script, extra in jobs:
            output = out_root / dataset / job_name
            command = [sys.executable, str(script), "--config", str(config), "--data-root", args.data_root,
                       "--out-dir", str(output), "--device", args.device, "--seed", str(args.seed), *extra]
            if args.quick:
                command.extend(["--quick-epochs", "1"])
            elif args.epochs is not None:
                command.extend(["--epochs", str(args.epochs)])
            print(f"\n[{dataset}/{job_name}] {' '.join(command)}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode:
                failures.append({"dataset": dataset, "job": job_name, "returncode": completed.returncode})
                if args.fail_fast:
                    break
                continue
            with (output / "summary_scores.csv").open(newline="", encoding="utf-8") as stream:
                rows.extend(csv.DictReader(stream))
        if failures and args.fail_fast:
            break
    if rows:
        with (out_root / "all_summary_scores.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if failures:
        with (out_root / "failures.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["dataset", "job", "returncode"])
            writer.writeheader()
            writer.writerows(failures)
    print(f"\nCompleted: {len(rows)} score rows; failures: {len(failures)}; output: {out_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
