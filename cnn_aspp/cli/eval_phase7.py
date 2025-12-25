# cnn_aspp/cli/eval_phase7.py

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_threshold_sweep(path: Path) -> pd.DataFrame:
    """
    Expects a CSV with at least the columns:
    - threshold
    - precision
    - recall
    - f1

    If your column names differ, tweak them below.
    """
    df = pd.read_csv(path)

    # Try to normalize column names a bit
    cols = {c.lower(): c for c in df.columns}
    # Required logical keys
    thr_col = cols.get("threshold", None)
    prec_col = cols.get("precision", None)
    rec_col = cols.get("recall", None)
    f1_col = cols.get("f1", None)

    missing = [k for k, v in
               [("threshold", thr_col), ("precision", prec_col),
                ("recall", rec_col), ("f1", f1_col)]
               if v is None]
    if missing:
        raise ValueError(
            f"threshold_sweep.csv is missing required columns: {missing}. "
            f"Got columns = {list(df.columns)}"
        )

    df = df[[thr_col, prec_col, rec_col, f1_col]].copy()
    df.columns = ["threshold", "precision", "recall", "f1"]
    return df


def plot_pr_curve(df: pd.DataFrame, out_path: Path) -> None:
    """
    Simple P–R curve with threshold annotations.
    """
    recalls = df["recall"].values
    precisions = df["precision"].values
    thresholds = df["threshold"].values

    plt.figure()
    plt.plot(recalls, precisions, marker="o")

    for r, p, t in zip(recalls, precisions, thresholds):
        plt.annotate(f"{t:.2f}", (r, p), fontsize=8)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall vs Threshold")
    plt.grid(True)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_best_threshold(df: pd.DataFrame, out_path: Path) -> dict:
    """
    Find the row with best F1 and write a small text summary.
    """
    best_row = df.loc[df["f1"].idxmax()]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        f.write(f"best_threshold={best_row['threshold']:.3f}\n")
        f.write(f"precision={best_row['precision']:.4f}\n")
        f.write(f"recall={best_row['recall']:.4f}\n")
        f.write(f"f1={best_row['f1']:.4f}\n")

    return {
        "threshold": float(best_row["threshold"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1": float(best_row["f1"]),
    }


def write_findings_md(best: dict, out_path: Path) -> None:
    """
    Minimal Findings.md you can expand with comments later.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        f.write("# Evaluation Findings (Phase 7)\n\n")
        f.write("## Global Threshold Sweep\n\n")
        f.write(
            "- Best F1 threshold: **{thr:.2f}**\n"
            "- Precision @ best: **{p:.3f}**\n"
            "- Recall @ best: **{r:.3f}**\n"
            "- F1 @ best: **{f1:.3f}**\n\n".format(
                thr=best["threshold"],
                p=best["precision"],
                r=best["recall"],
                f1=best["f1"],
            )
        )
        f.write("See `eval.csv` and `pr_curve.png` for full threshold sweep.\n\n")

        f.write("## Calibration\n\n")
        f.write(
            "- This run uses raw model probabilities (no Platt / isotonic "
            "calibration yet). Calibration can be added later by re-running "
            "evaluation on stored per-pixel probabilities.\n\n"
        )

        f.write("## Stratified Performance (TODO)\n\n")
        f.write(
            "- Planned stratification by ecoregion, fire-size bin, and "
            "vegetation cover once per-tile metadata is wired through "
            "the evaluation script.\n"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: turn threshold_sweep.csv into eval artifacts."
    )
    parser.add_argument(
        "--sweep_csv",
        type=str,
        default="lightning_logs/threshold_sweep.csv",
        help="Path to threshold_sweep.csv produced by sweep_thresh/scan_thresh.sh",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="cnn_aspp/ndws_out/eval",
        help="Directory to write Phase 7 artifacts into.",
    )

    args = parser.parse_args()
    sweep_path = Path(args.sweep_csv)
    out_dir = Path(args.out_dir)

    if not sweep_path.exists():
        raise SystemExit(
            f"Could not find sweep CSV at {sweep_path}. "
            f"Run scripts/scan_thresh.sh or your sweep script first."
        )

    print(f"Loading threshold sweep from: {sweep_path}")
    df = load_threshold_sweep(sweep_path)

    # 1) Save eval.csv
    eval_csv_path = out_dir / "eval.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(eval_csv_path, index=False)
    print(f"Wrote {eval_csv_path}")

    # 2) PR curve
    pr_path = out_dir / "pr_curve.png"
    plot_pr_curve(df, pr_path)
    print(f"Wrote {pr_path}")

    # 3) Best threshold
    best_txt_path = out_dir / "best_threshold.txt"
    best = write_best_threshold(df, best_txt_path)
    print(f"Wrote {best_txt_path} (best threshold = {best['threshold']:.3f})")

    # 4) Findings.md skeleton
    findings_path = out_dir / "Findings.md"
    write_findings_md(best, findings_path)
    print(f"Wrote {findings_path}")

    print("Phase 7 eval artifacts complete.")


if __name__ == "__main__":
    main()
