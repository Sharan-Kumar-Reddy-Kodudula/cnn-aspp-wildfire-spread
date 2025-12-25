# cnn_aspp/tasks/seg_task.py
from __future__ import annotations
import torch
import torch.nn as nn
import pytorch_lightning as pl
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from torchmetrics.classification import (
    BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryJaccardIndex
)

class TverskyLoss(nn.Module):
    """Tversky loss (1 – TI) for class-imbalanced segmentation."""
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.eps = alpha, beta, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N,1,H,W) ; targets: (N,H,W) or (N,1,H,W)
        probs = torch.sigmoid(logits)
        if targets.dim() == probs.dim() - 1:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        tp = (probs * targets).sum((2, 3))
        fp = (probs * (1 - targets)).sum((2, 3))
        fn = ((1 - probs) * targets).sum((2, 3))
        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1 - ti.mean()

@dataclass
class OptimCfg:
    name: str = "adam"
    lr: float = 4e-4
    weight_decay: float = 0.0

@dataclass
class SchedCfg:
    kind: str = "none"   # "none" | "cosine" | "step"
    t_max: int = 50
    eta_min: float = 1e-6
    step_size: int = 30
    gamma: float = 0.1

def _extract_xy(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Robustly extract (x, y) from common batch shapes:
    - (x, y) or (x, y, meta/weights/...)
    - dicts: {'x'/'image', 'y'/'mask'/...}
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        elif len(batch) == 1:
            batch = batch[0]
    if isinstance(batch, dict):
        x_keys = ("x", "inputs", "input", "image", "images")
        y_keys = ("y", "target", "targets", "mask", "masks", "label", "labels")
        x = next((batch[k] for k in x_keys if k in batch), None)
        y = next((batch[k] for k in y_keys if k in batch), None)
        if x is not None and y is not None:
            return x, y
    raise ValueError("Could not unpack batch into (x, y).")

class SegTrainTask(pl.LightningModule):
    """Lightning task: logs TI loss + Precision/Recall/F1/IoU with configurable threshold."""
    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.5,
        alpha: float = 0.5,
        beta: float = 0.5,
        optim: Dict[str, Any] | None = None,
        sched: Dict[str, Any] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = TverskyLoss(alpha, beta)
        self.prec = BinaryPrecision(threshold=threshold)
        self.rec = BinaryRecall(threshold=threshold)
        self.f1 = BinaryF1Score(threshold=threshold)
        self.iou = BinaryJaccardIndex(threshold=threshold)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _normalize_xy_shapes(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ensure x:(N,C,H,W); y:(N,H,W) int
        if y.dim() == 4 and y.size(1) == 1:
            y = y[:, 0]
        if y.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
            y = y.long()
        return x, y

    def _shared_step(self, batch, stage: str):
        x, y = _extract_xy(batch)
        x, y = self._normalize_xy_shapes(x, y)

        logits = self(x)
        loss = self.loss_fn(logits, y)

        probs = torch.sigmoid(logits).detach()
        if probs.dim() == 4 and probs.size(1) == 1:
            probs = probs[:, 0]

        bs = x.size(0)
        self.log(f"{stage}/ti_loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/precision", self.prec(probs, y.int()), on_epoch=True, batch_size=bs)
        self.log(f"{stage}/recall",   self.rec(probs, y.int()), on_epoch=True, batch_size=bs)
        self.log(f"{stage}/f1",       self.f1(probs, y.int()), prog_bar=True, on_epoch=True, batch_size=bs)
        self.log(f"{stage}/iou",      self.iou(probs, y.int()), prog_bar=True, on_epoch=True, batch_size=bs)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        ocfg = OptimCfg(**self.hparams.get("optim", {}))
        scfg = SchedCfg(**self.hparams.get("sched", {}))

        if ocfg.name.lower() == "adam":
            opt = torch.optim.Adam(self.parameters(), lr=ocfg.lr, weight_decay=ocfg.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer {ocfg.name}")

        if scfg.kind == "none":
            return opt
        if scfg.kind == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, scfg.t_max, scfg.eta_min)
        elif scfg.kind == "step":
            sch = torch.optim.lr_scheduler.StepLR(opt, scfg.step_size, scfg.gamma)
        else:
            raise ValueError(f"Unknown scheduler {scfg.kind}")
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
