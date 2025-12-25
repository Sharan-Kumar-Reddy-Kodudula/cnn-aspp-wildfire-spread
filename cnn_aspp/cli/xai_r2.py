# cnn_aspp/cli/xai_r2.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from cnn_aspp.xai.gradcam import GradCAM, upsample_to_match
from cnn_aspp.models.aspp_tiny import ASPPTiny


# ------------- NPZTileDataset (same as in xai_gradcam) -------------


class NPZTileDataset(Dataset):
    """
    Minimal dataset for micro NPZ tiles, reused for quantitative XAI.

    Expects:
        root/
          train/
            *.npz
          val/
            *.npz
    """

    def __init__(self, root: Path, split: str = "val"):
        super().__init__()
        self.root = Path(root)
        self.split = split
        split_dir = self.root / split
        self.files = sorted(split_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No NPZ files found under {split_dir}")

        print(f"[XAI] NPZTileDataset: found {len(self.files)} tiles in {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = np.load(path)

        keys = list(data.keys())
        if not keys:
            raise KeyError(f"{path} has no arrays inside")

        # --- pick image / input array ---
        x = None
        for k in ("x", "image", "inputs", "input"):
            if k in data:
                x = data[k]
                break
        if x is None:
            x = data[keys[0]]

        # --- pick mask / label array ---
        y = None
        for k in ("y", "mask", "target", "targets", "label", "labels"):
            if k in data:
                y = data[k]
                break
        if y is None:
            if len(keys) >= 2:
                y = data[keys[1]]
            else:
                y = data[keys[-1]]

        x = torch.from_numpy(x).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected x shape {x.shape} in {path}")

        y = torch.from_numpy(y)
        if y.ndim == 2:
            y = y.unsqueeze(0)
        elif y.ndim == 3:
            if y.shape[0] != 1:
                y = y[:1]
        else:
            raise ValueError(f"Unexpected y shape {y.shape} in {path}")
        y = y.float()

        tile_id = path.stem
        return {
            "image": x,
            "mask": y,
            "tile_id": tile_id,
        }


@dataclass
class XAIR2Config:
    ckpt_path: Path
    out_dir: Path
    data_root: Path
    split: str
    batch_size: int
    num_workers: int
    num_tiles: int
    device: str


def parse_args() -> XAIR2Config:
    parser = argparse.ArgumentParser(
        description="Phase 8B: Tiny quantitative XAI (R^2 between ASPP and final Grad-CAM, micro NPZ tiles)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to ASPPTiny Lightning checkpoint.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for XAI CSVs.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root of micro NPZ dataset (e.g. data/micro).",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--num_tiles",
        type=int,
        default=100,
        help="Number of validation tiles for R^2 computation.",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    return XAIR2Config(
        ckpt_path=Path(args.ckpt),
        out_dir=Path(args.out_dir),
        data_root=Path(args.data_root),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_tiles=args.num_tiles,
        device=args.device,
    )


# ----------------- helpers -----------------


def load_aspp_tiny_from_ckpt(ckpt_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    in_ch = None
    for k, v in state_dict.items():
        if k.endswith("stem1.conv.weight"):
            in_ch = v.shape[1]
            break
    if in_ch is None:
        print("[XAI] Could not infer in_channels from checkpoint; defaulting to 12.")
        in_ch = 12
    else:
        print(f"[XAI] Inferred in_channels={in_ch} from checkpoint.")

    model = ASPPTiny(in_channels=in_ch)

    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v
        else:
            new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print("[XAI] Loaded ASPPTiny from checkpoint.")
    print("      Missing keys:", missing)
    print("      Unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model


def build_val_dataset(data_root: Path, split: str) -> NPZTileDataset:
    return NPZTileDataset(root=data_root, split=split)


def build_eval_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def r2_score_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simple R^2 implementation: 1 - SS_res / SS_tot
    """
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ------------------------------ main ------------------------------


def main():
    cfg = parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[XAI] Using device: {device}")

    model = load_aspp_tiny_from_ckpt(cfg.ckpt_path, device=device)

    if not hasattr(model, "aspp_branches") or not hasattr(model, "final_conv"):
        raise AttributeError(
            "ASPPTiny must expose `aspp_branches` with keys 'd1','d3','d6','d12' "
            "and `final_conv` (see ASPPTiny implementation)."
        )

    aspp_branches: Dict[str, nn.Module] = model.aspp_branches
    final_conv: nn.Module = model.final_conv

    dataset = build_val_dataset(cfg.data_root, cfg.split)
    n = len(dataset)
    max_tiles = min(cfg.num_tiles, n)
    indices = list(range(max_tiles))
    subset = Subset(dataset, indices)
    loader = build_eval_loader(subset, cfg.batch_size, cfg.num_workers)

    gc_final = GradCAM(model, final_conv, device=device)
    gc_branches = {
        "d1": GradCAM(model, aspp_branches["d1"], device=device),
        "d3": GradCAM(model, aspp_branches["d3"], device=device),
        "d6": GradCAM(model, aspp_branches["d6"], device=device),
        "d12": GradCAM(model, aspp_branches["d12"], device=device),
    }

    rows = []
    tile_counter = 0

    for batch in loader:
        img: torch.Tensor = batch["image"]
        tile_ids = batch["tile_id"]

        img = img.to(device)

        with torch.set_grad_enabled(True):
            logits = model(img)  # [B,1,H,W]

        gc_out_final = gc_final(img, target_mask=None)
        cam_final = upsample_to_match(gc_out_final.cam, logits)  # [B,1,H,W]

        cam_branches = {}
        for key, gc in gc_branches.items():
            out_d = gc(img, target_mask=None)
            cam_branches[key] = upsample_to_match(out_d.cam, logits)

        B = img.size(0)
        for b in range(B):
            if tile_counter >= max_tiles:
                break

            tile_id = tile_ids[b]
            cam_final_np = cam_final[b].detach().cpu().numpy()[0].reshape(-1)

            for key, cam_d in cam_branches.items():
                dilation = int(key[1:])  # 'd1' -> 1, etc.
                cam_d_np = cam_d[b].detach().cpu().numpy()[0].reshape(-1)

                y_vec = cam_final_np
                x_vec = cam_d_np

                r2 = r2_score_numpy(y_vec, x_vec)
                rows.append(
                    {
                        "tile_idx": int(tile_counter),
                        "tile_id": str(tile_id),
                        "dilation": dilation,
                        "r2": r2,
                    }
                )

            tile_counter += 1

        if tile_counter >= max_tiles:
            break

    df = pd.DataFrame(rows)
    per_tile_path = cfg.out_dir / "xai_r2_per_tile.csv"
    df.to_csv(per_tile_path, index=False)

    summary = (
        df.groupby("dilation")["r2"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .sort_values("dilation")
    )

    summary_path = cfg.out_dir / "xai_r2_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("[XAI] Per-tile R^2 saved to:", per_tile_path)
    print("[XAI] Summary R^2 saved to:", summary_path)
    print(summary)


if __name__ == "__main__":
    main()
