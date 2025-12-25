# cnn_aspp/cli/reproduce_run.py
"""
Phase 9: Reproduce a reported score from a snapshot.

Usage (from repo root):

    python -m cnn_aspp.cli.reproduce_run \
      --snapshot runs/aspp_tiny_v68

IMPORTANT:
    Edit TRAIN_CMD below to match the EXACT training command you use
    for the run you want to reproduce (minus the leading 'python').
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd


# ---------------------------------------------------------------------------
# EDIT THIS for your project.
# This is the training command you normally run, WITHOUT the leading "python".
# Example: python -m cnn_aspp.cli.train_aspp_tiny dataset.split=micro ...
# becomes:
#     ["-m", "cnn_aspp.cli.train_aspp_tiny", "dataset.split=micro", ...]
# ---------------------------------------------------------------------------
TRAIN_CMD = [
    "-m",
    "cnn_aspp.cli.train",
    "dataset.split=micro",
    "task.threshold=0.05",
    "task.epochs=10",
    "task.amp=false",
]
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--snapshot",
        required=True,
        help="Path to runs/<name> snapshot directory (with stats.json)",
    )
    p.add_argument(
        "--logs_root",
        default="lightning_logs",
        help="Directory where Lightning writes version_*/",
    )
    p.add_argument(
        "--metric",
        default="val_f1",
        help="Metric name in metrics.csv to compare (default: val_f1)",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=0.5,
        help="Allowed absolute difference in metric (default: 0.5 F1 points)",
    )
    return p.parse_args()


def find_latest_version(logs_root: Path) -> Path | None:
    """Return the newest lightning_logs/version_xx directory."""
    if not logs_root.exists():
        return None
    version_dirs = [
        p for p in logs_root.iterdir()
        if p.is_dir() and p.name.startswith("version_")
    ]
    if not version_dirs:
        return None
    version_dirs.sort(key=os.path.getmtime)
    return version_dirs[-1]


def get_metric_from_run(version_dir: Path, metric_name: str) -> float | None:
    metrics_csv = version_dir / "metrics.csv"
    if not metrics_csv.exists():
        return None
    df = pd.read_csv(metrics_csv)
    if metric_name not in df.columns:
        # try alt name "val/f1" if metric_name == "val_f1"
        if metric_name == "val_f1" and "val/f1" in df.columns:
            metric_name = "val/f1"
        else:
            return None
    s = df[metric_name].dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def main() -> None:
    args = parse_args()
    snapshot = Path(args.snapshot).resolve()
    logs_root = Path(args.logs_root).resolve()

    stats_file = snapshot / "stats.json"
    if not stats_file.exists():
        raise SystemExit(f"stats.json not found in snapshot: {stats_file}")

    with open(stats_file) as f:
        stats = json.load(f)

    target_metric = stats.get(args.metric)
    if target_metric is None:
        raise SystemExit(
            f"{args.metric} is not present in stats.json. "
            f"Available keys: {list(stats.keys())}"
        )

    print(f"[reproduce_run] Target {args.metric} from snapshot: {target_metric:.4f}")

    # Find the latest run BEFORE retraining
    before = find_latest_version(logs_root)
    before_name = before.name if before is not None else "None"
    print(f"[reproduce_run] Latest version before retrain: {before_name}")

    # Build the full command: python + TRAIN_CMD
    cmd = [sys.executable] + TRAIN_CMD
    print("[reproduce_run] Running training command:")
    print("  ", " ".join(cmd))

    # Run training
    subprocess.check_call(cmd)

    # Find the latest run AFTER retraining
    after = find_latest_version(logs_root)
    if after is None:
        raise SystemExit("No lightning_logs/version_xx directories found after retraining.")

    if before is not None and after.samefile(before):
        print(
            "[reproduce_run] WARNING: no new version directory was created; "
            "did your training script reuse an existing version?"
        )

    print(f"[reproduce_run] Latest version after retrain: {after.name}")

    # Read metric from the reproduced run
    repro_metric = get_metric_from_run(after, args.metric)
    if repro_metric is None:
        raise SystemExit(
            f"Could not read {args.metric} from {after}/metrics.csv"
        )

    print(f"[reproduce_run] Reproduced {args.metric}: {repro_metric:.4f}")

    diff = abs(repro_metric - target_metric)
    print(f"[reproduce_run] |Δ{args.metric}| = {diff:.4f}")

    if diff <= args.tol:
        print(
            f"[reproduce_run] SUCCESS: {args.metric} within ±{args.tol} "
            f"of snapshot value."
        )
        sys.exit(0)
    else:
        print(
            f"[reproduce_run] WARNING: {args.metric} NOT within ±{args.tol} "
            f"of snapshot value."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
