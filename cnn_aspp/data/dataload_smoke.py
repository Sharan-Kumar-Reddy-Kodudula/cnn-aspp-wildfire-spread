# cnn_aspp/data/dataload_smoke.py
from __future__ import annotations
import argparse, os, random
from pathlib import Path
import numpy as np

def list_npz(root: Path):
    files = list(root.rglob("*.npz"))
    files.sort()
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Folder with tiles (e.g., NDWS_ROOT/train)")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.root)
    files = list_npz(root)
    if not files:
        raise SystemExit(f"No .npz files under {root}")

    bs = max(1, args.batch_size)
    picks = random.sample(files, min(bs, len(files)))
    xs, ys = [], []
    for p in picks:
        arr = np.load(p)
        xs.append(arr["inputs"])   # [C,H,W]
        ys.append(arr["targets"])  # [1,H,W]

    # stack
    X = np.stack(xs, axis=0)  # [B,C,H,W]
    Y = np.stack(ys, axis=0)  # [B,1,H,W]
    print(f"inputs: {list(X.shape)}  targets: {list(Y.shape)}")

    # class balance on the batch
    yflat = Y.reshape(-1)
    total = int(yflat.size)
    pos = int((yflat == 1).sum())
    print(f"class balance: fire=1 -> {pos}/{total} ({100.0*pos/max(1,total):.2f}%)")

if __name__ == "__main__":
    main()
