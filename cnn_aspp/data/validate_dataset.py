# cnn_aspp/data/validate_dataset.py
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter

def iter_npz(root: Path):
    for p in root.rglob("*.npz"):
        yield p

def maybe_read_meta(npz_path: Path):
    meta_path = npz_path.with_suffix(".json")
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="NDWS root folder with train/val/test")
    ap.add_argument("--tile-size", type=int, default=64)
    ap.add_argument("--expect-crs", default=None, help="e.g. EPSG:4326 (skip if None)")
    ap.add_argument("--out", required=True, help="Where to write stats.json")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"[ERR] root not found: {root}", file=sys.stderr)
        sys.exit(2)

    n_files = 0
    n_nan = 0
    n_inf = 0
    bad_tile = 0
    bad_crs = 0
    class_counts = Counter()

    # running stats per channel (Welford)
    chan_n = defaultdict(int)
    chan_mean = defaultdict(float)
    chan_M2 = defaultdict(float)
    chan_min = defaultdict(lambda: np.inf)
    chan_max = defaultdict(lambda: -np.inf)

    for npz_path in iter_npz(root):
        n_files += 1
        arr = np.load(npz_path)
        x = arr["inputs"]   # [C,H,W]
        y = arr["targets"]  # [1,H,W]
        C, H, W = x.shape

        # shape check
        if args.tile_size and (H != args.tile_size or W != args.tile_size):
            bad_tile += 1

        # CRS check (best effort)
        meta = maybe_read_meta(npz_path)
        crs = meta.get("crs")
        if args.expect_crs and crs and crs != args.expect_crs:
            bad_crs += 1

        # NaN / Inf check
        n_nan += int(np.isnan(x).any() or np.isnan(y).any())
        n_inf += int(np.isinf(x).any() or np.isinf(y).any())

        # class balance (0/1 in fire mask)
        # Note: ignore any values not 0/1 if present
        yy = y.reshape(-1)
        class_counts.update(yy.tolist())

        # per-channel stats
        # flatten per channel to update Welford stats
        for c in range(C):
            v = x[c].reshape(-1)
            # update min/max
            m = v.min()
            M = v.max()
            if m < chan_min[c]: chan_min[c] = float(m)
            if M > chan_max[c]: chan_max[c] = float(M)
            # Welford
            for val in v:
                chan_n[c] += 1
                delta = val - chan_mean[c]
                chan_mean[c] += delta / chan_n[c]
                chan_M2[c] += delta * (val - chan_mean[c])

    # finalize stats
    stats = {
        "num_files": n_files,
        "tile_size_expected": args.tile_size,
        "tile_size_mismatch_files": bad_tile,
        "crs_expected": args.expect_crs,
        "crs_mismatch_files": bad_crs,
        "has_nan_files": n_nan,
        "has_inf_files": n_inf,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "channels": {},
    }
    for c in sorted(chan_n.keys()):
        n = chan_n[c]
        mean = chan_mean[c]
        var = (chan_M2[c] / (n - 1)) if n > 1 else 0.0
        std = float(np.sqrt(var))
        stats["channels"][str(c)] = {
            "count": n,
            "mean": float(mean),
            "std": std,
            "min": float(chan_min[c]),
            "max": float(chan_max[c]),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] wrote stats to {out_path}")
    print(f"Files: {n_files} | tile_mismatch: {bad_tile} | crs_mismatch: {bad_crs} | NaN files: {n_nan} | Inf files: {n_inf}")
    if class_counts:
        total = sum(class_counts.values())
        pct1 = 100.0 * class_counts.get(1, 0) / max(total, 1)
        print(f"Class balance (mask==1): {pct1:.2f}% ({class_counts.get(1,0)}/{total})")

if __name__ == "__main__":
    main()
