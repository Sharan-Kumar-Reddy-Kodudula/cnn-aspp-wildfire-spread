# cnn_aspp/cli/sweep_thresh.py
from __future__ import annotations

import os
import csv
import math
import inspect
from pathlib import Path
from typing import List, Optional, Tuple, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import hydra

from cnn_aspp.models.aspp_tiny import ASPPTiny
from cnn_aspp.data.ndws_dataset import NDWSTilesDataset


# =========================
# Model constructor
# =========================
def _build_aspp_tiny(cfg_model) -> ASPPTiny:
    """
    Build ASPPTiny from a generic Hydra model config.
    Only uses fields that exist in ASPPTiny.__init__.
    """
    sig = inspect.signature(ASPPTiny.__init__)
    params = sig.parameters
    kw = {}

    # in_channels
    if "in_channels" in params:
        kw["in_channels"] = cfg_model.in_channels
    elif "in_ch" in params:
        kw["in_ch"] = cfg_model.in_channels

    # width / filters
    width_val = cfg_model.filters
    for alias in ("filters", "width", "base_channels", "base_width",
                  "features", "planes", "channels"):
        if alias in params:
            kw[alias] = width_val
            break

    # ASPP dilations
    rates_val = list(cfg_model.dilations)
    for alias in ("dilations", "rates", "atrous_rates", "aspp_rates"):
        if alias in params:
            kw[alias] = rates_val
            break

    # binary out (1 channel)
    for alias in ("out_channels", "out_ch", "num_classes"):
        if alias in params:
            kw[alias] = 1
            break

    # Filter only valid kwargs
    kw = {k: v for k, v in kw.items() if k in params}
    return ASPPTiny(**kw)


# =========================
# Robust batch parsing
# =========================
COMMON_X_KEYS = ("x", "image", "inputs", "feat", "features", "data")
COMMON_Y_KEYS = ("y", "mask", "target", "label", "labels",
                 "y_true", "gt", "masks")


def _is_tensor(obj: Any) -> bool:
    return isinstance(obj, Tensor)


def _maybe_squeeze_y(y: Tensor) -> Tensor:
    # (N,1,H,W) -> (N,H,W)
    if y.dim() == 4 and y.size(1) == 1:
        return y[:, 0]
    return y


def _pick_x_y_from_sequence(seq) -> Optional[Tuple[Tensor, Tensor]]:
    tensors = [t for t in seq if _is_tensor(t)]
    if len(tensors) >= 2:
        return tensors[0], _maybe_squeeze_y(tensors[1])

    for item in seq:
        if isinstance(item, (list, tuple)):
            out = _pick_x_y_from_sequence(item)
            if out:
                return out
        if isinstance(item, dict):
            out = _pick_x_y_from_mapping(item)
            if out:
                return out
    return None


def _pick_x_y_from_mapping(mp: dict) -> Optional[Tuple[Tensor, Tensor]]:
    x = None
    y = None
    for k in COMMON_X_KEYS:
        if k in mp and _is_tensor(mp[k]):
            x = mp[k]
            break
    for k in COMMON_Y_KEYS:
        if k in mp and _is_tensor(mp[k]):
            y = mp[k]
            break
    if x is not None and y is not None:
        return x, _maybe_squeeze_y(y)

    # Heuristic fallback
    tens = [v for v in mp.values() if _is_tensor(v)]
    if not tens:
        return None

    x_cands = sorted(
        [t for t in tens if t.dtype.is_floating_point],
        key=lambda t: (t.numel(), t.dim()),
        reverse=True,
    )
    if x_cands:
        x0 = x_cands[0]
        x_spatial = x0.shape[-2:]
        y_cands = []
        for v in tens:
            if v is x0:
                continue
            if (
                (not v.dtype.is_floating_point)
                or (v.shape[-2:] == x_spatial)
                or (v.dim() == 3 and v.shape[0] == x0.shape[0])
            ):
                y_cands.append(v)

        if y_cands:
            return x0, _maybe_squeeze_y(y_cands[0])
        return x0, _maybe_squeeze_y(tens[0])

    return None


def _smart_xy_from_any(batch: Any) -> Tuple[Tensor, Tensor]:
    if isinstance(batch, (list, tuple)):
        out = _pick_x_y_from_sequence(batch)
        if out:
            return out
    if isinstance(batch, dict):
        out = _pick_x_y_from_mapping(batch)
        if out:
            return out
    raise RuntimeError("Unexpected batch format; can't unpack (x, y).")


# =========================
# Metrics
# =========================
def _compute_pr_f1_iou(pred: Tensor, tgt: Tensor, eps: float = 1e-9):
    # pred,tgt: same shape (N,H,W)
    pred = pred.to(torch.bool)
    tgt = tgt.to(torch.bool)
    tp = (pred & tgt).sum().item()
    fp = (pred & ~tgt).sum().item()
    fn = (~pred & tgt).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return dict(precision=precision, recall=recall, f1=f1, iou=iou)


# =========================
# Collect logits & targets
# =========================
def _collect_logits_and_targets(
    model: torch.nn.Module, val_loader: DataLoader, device: torch.device
):
    model.eval().to(device)
    all_probs, all_tgts = [], []
    with torch.no_grad():
        for batch in val_loader:
            x, y = _smart_xy_from_any(batch)
            x = x.to(device)
            y = _maybe_squeeze_y(y).to(device)
            probs = torch.sigmoid(model(x))
            if probs.dim() == 4 and probs.size(1) == 1:
                probs = probs[:, 0]  # (N,H,W)
            all_probs.append(probs.detach().cpu())
            all_tgts.append(y.detach().cpu())

    if len(all_probs) == 0:
        raise RuntimeError("Validation loader yielded no batches.")
    return torch.cat(all_probs, 0), torch.cat(all_tgts, 0)


def _sweep(thresholds: List[float], probs: Tensor, tgts: Tensor):
    # probs,tgts: (N,H,W)
    rows = []
    for t in thresholds:
        pred = (probs >= t)
        m = _compute_pr_f1_iou(pred, tgts)
        rows.append(
            dict(
                threshold=float(round(t, 4)),
                **{k: float(v) for k, v in m.items()},
            )
        )
    return rows


def _maybe_load_ckpt(model: torch.nn.Module, ckpt_path: Optional[str]):
    if not ckpt_path:
        return model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    new_state = {
        (k[len("model.") :] if k.startswith("model.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(new_state, strict=False)
    return model


# =========================
# Entry
# =========================
@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # 1) Build model from config
    model = _build_aspp_tiny(cfg.model)

    # 2) Build val dataset & loader directly for NDWS
    #    Use dataset.root if set; otherwise fall back to the default in configs.
    root_cfg = getattr(cfg.dataset, "root", None)
    if root_cfg is None or str(root_cfg) == "":
        root_cfg = "./cnn_aspp/data/ndws_out"

    root = Path(str(root_cfg))
    val_dir = root / "val" / "EVTUNK"

    stats_path = getattr(cfg.dataset, "stats_path", "./cnn_aspp/data/stats.json")
    val_ds = NDWSTilesDataset(str(val_dir), stats_path)
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.task.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 3) Load checkpoint (hardcoded to your best full-NDWS run)
    ckpt_path = "tb/version_1/checkpoints/epochepoch=018-val_IoUval/IoU=0.261.ckpt"
    print(f"[sweep] Loading checkpoint: {ckpt_path}")
    _maybe_load_ckpt(model, ckpt_path)

    # 4) Collect probabilities and targets
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs, tgts = _collect_logits_and_targets(model, val_loader, device=device)

    # 5) Threshold range (from cfg or defaults)
    def _get(key: str, default: float) -> float:
        if key in cfg:
            return float(cfg.get(key))
        if "task" in cfg and key in cfg.task:
            return float(cfg.task.get(key))
        return default

    lo = _get("sweep_lo", 0.0)
    hi = _get("sweep_hi", 1.0)
    step = _get("sweep_step", 0.05)

    n = int(math.floor((hi - lo) / step)) + 1
    thresholds = [round(lo + i * step, 10) for i in range(max(n, 1))]

    # 6) Sweep
    rows = _sweep(thresholds, probs, tgts)

    print("\nthreshold  precision  recall  f1     iou")
    print("--------------------------------------------")
    for r in rows:
        print(
            f"{r['threshold']:>9.2f}  {r['precision']:>9.3f}  "
            f"{r['recall']:>6.3f}  {r['f1']:>5.3f}  {r['iou']:>5.3f}"
        )

    best_f1 = max(rows, key=lambda r: r["f1"])
    best_iou = max(rows, key=lambda r: r["iou"])
    print(
        "\n[best by F1]  t={threshold:.2f}  P={precision:.3f} "
        "R={recall:.3f} F1={f1:.3f} IoU={iou:.3f}".format(**best_f1)
    )
    print(
        "[best by IoU] t={threshold:.2f}  P={precision:.3f} "
        "R={recall:.3f} F1={f1:.3f} IoU={iou:.3f}".format(**best_iou)
    )

    # 7) Write CSV
    out_csv = os.path.join("./lightning_logs", "threshold_sweep.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["threshold", "precision", "recall", "f1", "iou"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[sweep] wrote {out_csv}")
    print("[sweep] Tip: set `task.threshold=<best>` for future runs.")


if __name__ == "__main__":
    main()
