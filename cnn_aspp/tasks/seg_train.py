# ---------------------------
# Lightning Task (training loop)
# File: cnn_aspp/tasks/seg_train.py
# ---------------------------
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score


@dataclass
class OptimCfg:
    lr: float = 3e-4
    weight_decay: float = 1e-4          # Phase 6: regularization
    betas: tuple = (0.9, 0.999)
    scheduler: str = "cosine"           # or "none"
    warmup_steps: int = 500


class SegLightning(pl.LightningModule):
    """
    Thin Lightning wrapper around a segmentation model that exposes:

      - model.compute_loss(logits, targets, mask, criterion=...)
        where criterion ∈ {"tversky", "bce", "focal"}

      - logs:
          train/loss, val/loss
          val/IoU (BinaryJaccardIndex)
          val/F1  (BinaryF1Score)

    The underlying model is expected to:
      * accept input [N,C,H,W]
      * return logits [N,1,H,W]
      * implement .compute_loss(...)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: str = "tversky",
        optim_cfg: OptimCfg = OptimCfg(),
        thresh: float = 0.5,
    ):
        super().__init__()
        # we don't save the model object itself in hyperparams to avoid bloat
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = criterion
        self.optim_cfg = optim_cfg
        self.thresh = float(thresh)

        # Metrics at threshold `thresh`
        self.iou = BinaryJaccardIndex()
        self.f1 = BinaryF1Score()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: Any, stage: str) -> torch.Tensor:
        """
        Accepts either:
          - dict batch: {"inputs": x, "targets": y, "mask": (optional) m}
          - tuple/list: (x, y) or (x, y, m)

        Shapes:
          x: [N, C, H, W]
          y: [N, H, W] or [N, 1, H, W]
          m: same spatial size as y (optional)
        """
        # --- Unpack batch flexibly ---
        if isinstance(batch, dict):
            x = batch["inputs"]
            y = batch["targets"]
            m = batch.get("mask", None)
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, y, m = batch
            elif len(batch) == 2:
                x, y = batch
                m = None
            else:
                raise ValueError(f"Unexpected batch tuple/list length: {len(batch)}")
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        # --- Forward + loss ---
        logits = self.model(x)
        # ASPPTiny.compute_loss supports: "tversky", "bce", "focal"
        loss = self.model.compute_loss(
            logits,
            y,
            m,
            criterion=self.criterion,
        )["loss"]

        # --- Metrics ---
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= self.thresh).float()
            y_m = y.unsqueeze(1) if y.dim() == 3 else y  # [N,1,H,W]

            self.iou.update(preds, y_m.int())
            self.f1.update(preds, y_m.int())

        # logging
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=x.shape[0],
        )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        iou = self.iou.compute()
        f1 = self.f1.compute()
        self.log("val/IoU", iou, prog_bar=True)
        self.log("val/F1", f1, prog_bar=True)
        self.iou.reset()
        self.f1.reset()

    # --- Optimizer / Scheduler ---
    def configure_optimizers(self):
        cfg = self.optim_cfg
        opt = optim.AdamW(
            self.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        if cfg.scheduler == "none":
            return opt

        # Cosine with linear warmup by step
        def lr_lambda(step: int):
            if step < cfg.warmup_steps:
                return max(1e-3, step / max(1, cfg.warmup_steps))
            total = max(1, self.trainer.estimated_stepping_batches - cfg.warmup_steps)
            progress = (step - cfg.warmup_steps) / total
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))

        sch = optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step"},
        }
