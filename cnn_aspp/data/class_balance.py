# cnn_aspp/data/class_balance.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def scan_split(split_dir: Path) -> dict:
    npzs = list(split_dir.rglob("*.npz"))
    fire = 0
    total = 0
    for p in npzs:
        arr = np.load(p)
        y = arr["targets"]  # [1,H,W]
        fire += int((y == 1).sum())
        total += int(y.size)
    return {
        "files": len(npzs),
        "fire_pixels": fire,
        "total_pixels": total,
        "fire_frac": (fire / total) if total else 0.0,
    }

def main():
    ap = argparse.ArgumentParser(description="Compute fire/non-fire class balance per split.")
    ap.add_argument("--root", required=True, help="NDWS root (with train/val/test)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    report = {}
    for split in ["train", "val", "test"]:
        d = root / split
        if d.exists():
            report[split] = scan_split(d)

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)
    print(f"Wrote {out}\n{json.dumps(report, indent=2)}")

if __name__ == "__main__":
    main()
