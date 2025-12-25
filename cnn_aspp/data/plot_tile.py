# cnn_aspp/data/plot_tile.py
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_channels_list(p: str | None, c: int) -> list[str]:
    if p and Path(p).exists():
        names = json.load(open(p))
        if len(names) >= c:
            return names[:c]
        # pad if shorter
        names = names + [f"ch{i}" for i in range(len(names), c)]
        return names
    return [f"ch{i}" for i in range(c)]

def main():
    ap = argparse.ArgumentParser(description="Visualize NDWS tile channels and mask.")
    ap.add_argument("--root", required=True, help="Split dir (e.g. .../ndws_out/train)")
    ap.add_argument("--channels-json", default=None, help="cnn_aspp/data/channels.json")
    ap.add_argument("--index", type=int, default=None, help="Specific tile index to show (sorted order).")
    ap.add_argument("--first-k", type=int, default=4, help="How many channels to show.")
    ap.add_argument("--overlay-mask", action="store_true", help="Overlay fire mask contour on each channel.")
    ap.add_argument("--save", default=None, help="Path to save PNG instead of showing (e.g., out.png).")
    args = ap.parse_args()

    files = sorted(Path(args.root).rglob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz tiles under {args.root}")

    idx = args.index if args.index is not None else random.randrange(len(files))
    p = files[min(max(idx, 0), len(files)-1)]
    arr = np.load(p)
    x, y = arr["inputs"], arr["targets"]  # x:[C,H,W], y:[1,H,W]

    C = x.shape[0]
    K = max(1, min(args.first_k, C))
    ch_names = load_channels_list(args.channels_json, C)

    # layout: K channels + 1 mask panel
    fig, axes = plt.subplots(1, K + 1, figsize=(3.6 * (K + 1), 3.6))
    axes = np.atleast_1d(axes)

    for i in range(K):
        im = axes[i].imshow(x[i], interpolation="nearest")
        axes[i].set_title(ch_names[i])
        axes[i].axis("off")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        if args.overlay_mask:
            # draw mask contour (value==1)
            axes[i].contour(y[0], levels=[0.5], linewidths=1.2)

    axes[K].imshow(y[0], interpolation="nearest")
    axes[K].set_title("FireMask")
    axes[K].axis("off")

    fig.suptitle(str(p))
    plt.tight_layout()

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150)
        print(f"Saved: {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
