# cnn_aspp/cli/xai_gradcam.py

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from torch.utils.data import DataLoader, Dataset, Subset

from cnn_aspp.xai.gradcam import GradCAM, upsample_to_match
from cnn_aspp.models.aspp_tiny import ASPPTiny


# ----------------- tiny NPZ dataset for micro split -----------------


class NPZTileDataset(Dataset):
    """
    Minimal dataset for micro NPZ tiles.

    Expects directory layout:
        root/
          train/
            *.npz
          val/
            *.npz

    Each .npz file should contain two arrays:
      - one for the input image (C,H,W or H,W)
      - one for the mask/label (H,W or [1,H,W])

    We try to guess keys; if unknown, we fall back to the first / second arrays.
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
            # fallback: first array
            x = data[keys[0]]

        # --- pick mask / label array ---
        y = None
        for k in ("y", "mask", "target", "targets", "label", "labels"):
            if k in data:
                y = data[k]
                break
        if y is None:
            # fallback: second array if it exists, otherwise reuse the last
            if len(keys) >= 2:
                y = data[keys[1]]
            else:
                y = data[keys[-1]]

        x = torch.from_numpy(x).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)  # [1,H,W]
        elif x.ndim == 3:
            # assume [C,H,W]
            pass
        else:
            raise ValueError(f"Unexpected x shape {x.shape} in {path}")

        y = torch.from_numpy(y)
        if y.ndim == 2:
            y = y.unsqueeze(0)  # [1,H,W]
        elif y.ndim == 3:
            # assume [1,H,W] or [C,H,W]; we only need a single channel
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


# ----------------- config & arg parsing -----------------


@dataclass
class XAIConfig:
    ckpt_path: Path
    out_dir: Path
    data_root: Path
    split: str
    batch_size: int
    num_workers: int
    num_tiles: int
    fp_fn_json: Optional[Path]
    device: str


def parse_args() -> XAIConfig:
    parser = argparse.ArgumentParser(
        description="Phase 8A: Grad-CAM qualitative XAI & error atlas (micro NPZ tiles)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to ASPPTiny Lightning checkpoint (e.g. tb/version_7/checkpoints/best.ckpt)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for XAI visuals.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root of micro NPZ dataset (e.g. data/micro).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split name (default: val).",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--num_tiles",
        type=int,
        default=40,
        help="Number of tiles to visualize.",
    )
    parser.add_argument(
        "--fp_fn_json",
        type=str,
        default=None,
        help="Optional JSON with per-tile FP/FN info from Phase 7 (not required for micro).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on ('cuda' or 'cpu').",
    )

    args = parser.parse_args()

    return XAIConfig(
        ckpt_path=Path(args.ckpt),
        out_dir=Path(args.out_dir),
        data_root=Path(args.data_root),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_tiles=args.num_tiles,
        fp_fn_json=Path(args.fp_fn_json) if args.fp_fn_json else None,
        device=args.device,
    )


# ----------------- model helpers -----------------


def load_aspp_tiny_from_ckpt(ckpt_path: Path, device: torch.device) -> nn.Module:
    """
    Load ASPPTiny from a Lightning checkpoint.

    - torch.load() the checkpoint.
    - Extract state_dict (ckpt['state_dict'] if present).
    - Infer in_channels from stem1.conv.weight shape.
    - Strip an optional 'model.' prefix on keys.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # Infer in_channels
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

    # Remove optional 'model.' prefix (common in Lightning)
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


def select_tiles(
    dataset,
    num_tiles: int,
    fp_fn_json: Optional[Path],
) -> List[int]:
    n = len(dataset)
    indices = list(range(n))

    # For micro, fp_fn_json is usually None; just grab first num_tiles.
    if fp_fn_json is None:
        return indices[: num_tiles]

    with fp_fn_json.open("r") as f:
        entries = json.load(f)

    selected = [int(e["idx"]) for e in entries if 0 <= int(e["idx"]) < n]
    if len(selected) > num_tiles:
        selected = selected[: num_tiles]

    if not selected:
        return indices[: num_tiles]

    return selected


# ----------------- visualization utils -----------------


def to_rgb_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert [C,H,W] to [H,W,3] in [0,1], using first 3 channels.
    """
    x = x.detach().cpu().float()
    C, H, W = x.shape

    if C >= 3:
        rgb = x[:3, :, :]
    else:
        rgb = x[0:1, :, :].repeat(3, 1, 1)

    rgb = rgb.view(3, -1)
    minv = rgb.min(dim=1, keepdim=True)[0]
    maxv = rgb.max(dim=1, keepdim=True)[0]
    rgb = (rgb - minv) / (maxv - minv + 1e-6)
    rgb = rgb.view(3, H, W)

    return rgb.permute(1, 2, 0).numpy()


def overlay_cam_on_image(
    base_img: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    cam = np.clip(cam, 0.0, 1.0)
    heatmap = cm.jet(cam)[..., :3]
    overlay = (1 - alpha) * base_img + alpha * heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def overlay_mask_outline(base_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Draw green outlines for mask on top of base_img using OpenCV.
    If anything goes wrong, just return the base image.
    """
    try:
        import cv2  # type: ignore
    except ImportError:
        return base_img

    try:
        mask = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )

        img = (base_img * 255).astype(np.uint8)
        img = np.ascontiguousarray(img)

        cv2.drawContours(img, contours, contourIdx=-1, color=(0, 255, 0), thickness=1)
        return img.astype(np.float32) / 255.0
    except Exception as e:
        print(f"[XAI] overlay_mask_outline: OpenCV failed with {e}; returning base image.")
        return base_img


# ------------------------------ main ------------------------------


def main():
    cfg = parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    qual_dir = cfg.out_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[XAI] Using device: {device}")

    model = load_aspp_tiny_from_ckpt(cfg.ckpt_path, device=device)

    if not hasattr(model, "aspp_branches") or not hasattr(model, "final_conv"):
        raise AttributeError(
            "ASPPTiny must expose `aspp_branches` (ModuleDict with 'd1','d3','d6','d12') "
            "and `final_conv` (last conv layer)."
        )

    aspp_branches: Dict[str, nn.Module] = model.aspp_branches
    final_conv: nn.Module = model.final_conv

    dataset = build_val_dataset(cfg.data_root, cfg.split)
    selected_indices = select_tiles(dataset, cfg.num_tiles, cfg.fp_fn_json)
    subset = Subset(dataset, selected_indices)
    loader = build_eval_loader(subset, cfg.batch_size, cfg.num_workers)

    gc_final = GradCAM(model, final_conv, device=device)
    gc_branches = {
        "d1": GradCAM(model, aspp_branches["d1"], device=device),
        "d3": GradCAM(model, aspp_branches["d3"], device=device),
        "d6": GradCAM(model, aspp_branches["d6"], device=device),
        "d12": GradCAM(model, aspp_branches["d12"], device=device),
    }

    html_entries: List[Tuple[str, str]] = []
    tile_counter = 0

    for batch in loader:
        img: torch.Tensor = batch["image"]  # [B,C,H,W]
        mask: torch.Tensor = batch["mask"]  # [B,1,H,W]
        tile_ids = batch["tile_id"]         # list/array of strings

        img = img.to(device)
        mask = mask.to(device)

        with torch.set_grad_enabled(True):
            logits = model(img)  # [B,1,H,W]
            probs = torch.sigmoid(logits)
        y_hat = (probs > 0.5).float()

        gc_out_final = gc_final(img, target_mask=None)
        cam_final = upsample_to_match(gc_out_final.cam, logits)

        cams_branches = {}
        for d_key, gc in gc_branches.items():
            out_d = gc(img, target_mask=None)
            cams_branches[d_key] = upsample_to_match(out_d.cam, logits)

        B = img.size(0)
        for b in range(B):
            if tile_counter >= cfg.num_tiles:
                break

            tile_id = str(tile_ids[b])
            x_b = img[b].detach().cpu()
            y_true_b = mask[b].detach().cpu()
            y_pred_b = y_hat[b].detach().cpu()

            base_img = to_rgb_image(x_b)
            mask_true_np = y_true_b.numpy()[0]
            mask_pred_np = y_pred_b.numpy()[0]

            cam_final_np = cam_final[b].detach().cpu().numpy()[0]
            cam_d1_np = cams_branches["d1"][b].detach().cpu().numpy()[0]
            cam_d3_np = cams_branches["d3"][b].detach().cpu().numpy()[0]
            cam_d6_np = cams_branches["d6"][b].detach().cpu().numpy()[0]
            cam_d12_np = cams_branches["d12"][b].detach().cpu().numpy()[0]

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            ax = axes[0, 0]
            ax.imshow(base_img)
            ax.set_title("Input")
            ax.axis("off")

            ax = axes[0, 1]
            gt_overlay = overlay_mask_outline(base_img, mask_true_np)
            ax.imshow(gt_overlay)
            ax.set_title("GT mask")
            ax.axis("off")

            ax = axes[0, 2]
            pred_overlay = overlay_mask_outline(base_img, mask_pred_np)
            ax.imshow(pred_overlay)
            ax.set_title("Pred mask")
            ax.axis("off")

            ax = axes[0, 3]
            final_overlay = overlay_cam_on_image(base_img, cam_final_np)
            ax.imshow(final_overlay)
            ax.set_title("Final conv Grad-CAM")
            ax.axis("off")

            ax = axes[1, 0]
            ax.imshow(overlay_cam_on_image(base_img, cam_d1_np))
            ax.set_title("ASPP d=1")
            ax.axis("off")

            ax = axes[1, 1]
            ax.imshow(overlay_cam_on_image(base_img, cam_d3_np))
            ax.set_title("ASPP d=3")
            ax.axis("off")

            ax = axes[1, 2]
            ax.imshow(overlay_cam_on_image(base_img, cam_d6_np))
            ax.set_title("ASPP d=6")
            ax.axis("off")

            ax = axes[1, 3]
            ax.imshow(overlay_cam_on_image(base_img, cam_d12_np))
            ax.set_title("ASPP d=12")
            ax.axis("off")

            fig.suptitle(f"Tile {tile_id}", fontsize=14)
            fig.tight_layout()

            out_name = f"tile_{tile_id}.png"
            out_path = qual_dir / out_name
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

            html_entries.append((tile_id, f"qualitative/{out_name}"))
            tile_counter += 1

            if tile_counter >= cfg.num_tiles:
                break

        if tile_counter >= cfg.num_tiles:
            break

    html_path = cfg.out_dir / "error_atlas.html"
    with html_path.open("w") as f:
        f.write("<html><body>\n")
        f.write("<h1>Wildfire CNN-ASPP Error Atlas (Grad-CAM, micro split)</h1>\n")
        for tile_id, rel_path in html_entries:
            f.write(f"<h2>Tile {tile_id}</h2>\n")
            f.write(f"<img src='{rel_path}' width='600'><br/>\n")
        f.write("</body></html>\n")

    print(f"[XAI] Qualitative Grad-CAM PNGs in: {qual_dir}")
    print(f"[XAI] Error atlas HTML: {html_path}")


if __name__ == "__main__":
    main()
