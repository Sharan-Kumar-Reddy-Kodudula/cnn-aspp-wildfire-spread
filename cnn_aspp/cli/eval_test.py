# cnn_aspp/cli/eval_test.py
from __future__ import annotations
import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnn_aspp.models.aspp_tiny import ASPPTiny
from cnn_aspp.data.ndws_dataset import NDWSTilesDataset


def iou_f1(pred: np.ndarray, targ: np.ndarray) -> tuple[float, float]:
    # pred, targ: [H,W] uint8 {0,1}
    inter = (pred & targ).sum()
    union = (pred | targ).sum()
    iou = inter / (union + 1e-9)
    f1 = (2 * inter) / (pred.sum() + targ.sum() + 1e-9)
    return float(iou), float(f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to Lightning/torch checkpoint")
    ap.add_argument("--in_channels", type=int, required=True)
    ap.add_argument("--root", required=True, help="Dataset split root (e.g., .../test)")
    ap.add_argument("--stats_path", default=None, help="stats.json used during training (for z-score)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs/test_preds", help="Where to save .npy preds")
    args = ap.parse_args()

    ds = NDWSTilesDataset(root=args.root, stats_path=args.stats_path)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    m = ASPPTiny(in_channels=args.in_channels).eval()
    state = torch.load(args.ckpt, map_location="cpu")
    sd = state.get("state_dict", state)
    m.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()
                       if k.startswith("model.") or k in m.state_dict()})

    os.makedirs(args.out_dir, exist_ok=True)

    ious, f1s = [], []
    idx = 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Evaluating"):
            x = batch["inputs"]              # [B,C,H,W]
            y = batch["targets"]             # [B,1,H,W], {0,1}
            p = m.predict(x)                 # [B,1,H,W] in [0,1]
            pred = (p >= args.threshold).to(torch.uint8).cpu().numpy()
            targ = (y > 0).to(torch.uint8).cpu().numpy()

            # save + metrics
            for b in range(pred.shape[0]):
                np.save(os.path.join(args.out_dir, f"pred_{idx:06d}.npy"), pred[b, 0])
                ii, ff = iou_f1(pred[b, 0], targ[b, 0])
                ious.append(ii); f1s.append(ff)
                idx += 1

    print(f"Test IoU: {np.mean(ious):.3f} ± {np.std(ious):.3f}")
    print(f"Test F1 : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")


if __name__ == "__main__":
    main()
