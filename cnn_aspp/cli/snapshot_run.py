# cnn_aspp/cli/snapshot_run.py
"""
Phase 9: Snapshot a finished Lightning run into runs/<name>.

Usage (from repo root):

    python -m cnn_aspp.cli.snapshot_run \
      --version_dir lightning_logs/version_68 \
      --out_dir runs/aspp_tiny_v68
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--version_dir",
        required=True,
        help="Path to a lightning_logs/version_xx directory",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Where to write the snapshot, e.g. runs/aspp_tiny_v68",
    )
    p.add_argument(
        "--project_root",
        default=".",
        help="Path to repo root (for copying cnn_aspp/conf and code)",
    )
    return p.parse_args()


def summarize_metrics(metrics_csv: Path) -> dict:
    df = pd.read_csv(metrics_csv)

    def last_any(cols):
        for c in cols:
            if c in df.columns:
                s = df[c].dropna()
                if not s.empty:
                    return float(s.iloc[-1])
        return None

    stats = {
        "val_f1": last_any(["val_f1", "val/f1"]),
        "val_precision": last_any(["val_precision", "val/precision"]),
        "val_recall": last_any(["val_recall", "val/recall"]),
        "val_oa": last_any(["val_oa"]),
    }

    # Best epoch by val_f1 if available
    if "val_f1" in df.columns:
        s = df["val_f1"].dropna()
        if not s.empty:
            best_idx = s.idxmax()
            stats["best_epoch"] = int(df.loc[best_idx, "epoch"])
    return stats


def snapshot_run(version_dir: Path, out_dir: Path, project_root: Path) -> None:
    version_dir = version_dir.resolve()
    out_dir = out_dir.resolve()
    project_root = project_root.resolve()

    if not version_dir.exists():
        raise SystemExit(f"version_dir does not exist: {version_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) metrics → stats.json
    metrics_csv = version_dir / "metrics.csv"
    if metrics_csv.exists():
        stats = summarize_metrics(metrics_csv)
    else:
        stats = {}
    stats["version_dir"] = str(version_dir)

    # Try to record git commit, if repo is a git repo.
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
            )
            .decode("utf-8")
            .strip()
        )
        stats["git_commit"] = commit
    except Exception:
        stats["git_commit"] = None

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # 2) Copy hparams.yaml (Lightning's hyperparams snapshot)
    hparams_src = version_dir / "hparams.yaml"
    if hparams_src.exists():
        shutil.copy2(hparams_src, out_dir / "hparams.yaml")

    # 3) Copy current conf/ for clarity of training setup
    conf_src = project_root / "cnn_aspp" / "conf"
    if conf_src.exists():
        shutil.copytree(
            conf_src,
            out_dir / "conf",
            dirs_exist_ok=True,
        )

    # 4) Freeze environment
    try:
        req_out = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
        with open(out_dir / "requirements.txt", "w") as f:
            f.write(req_out)
    except Exception as e:
        print(f"[snapshot_run] WARNING: could not run 'pip freeze': {e}")

    # 5) Minimal code snapshot
    code_src = project_root / "cnn_aspp"
    code_dst = out_dir / "code" / "cnn_aspp"
    if code_src.exists():
        shutil.copytree(
            code_src,
            code_dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                "__pycache__",
                "*.pyc",
                "*.pyo",
                "*.pyd",
                "lightning_logs",
                "tb",
                "runs",
                "ndws_out",
                "outputs",
                "data",
            ),
        )

    print(f"[snapshot_run] Snapshot written to: {out_dir}")


def main() -> None:
    args = parse_args()
    snapshot_run(Path(args.version_dir), Path(args.out_dir), Path(args.project_root))


if __name__ == "__main__":
    main()
