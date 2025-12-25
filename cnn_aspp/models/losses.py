# cnn_aspp/models/losses.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """Plain BCE with logits for segmentation."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B,1,H,W]
            targets: [B,1,H,W] in {0,1}
        """
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction=self.reduction
        )


class TverskyLoss(nn.Module):
    """
    Binary Tversky loss for segmentation with heavy class imbalance.

    Args:
        alpha: weight for false positives
        beta: weight for false negatives
        eps: numerical stability constant
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B,1,H,W]
            targets: [B,1,H,W] in {0,1}
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        dims = (0, 2, 3)  # sum over batch + spatial
        tp = torch.sum(probs * targets, dim=dims)
        fp = torch.sum(probs * (1.0 - targets), dim=dims)
        fn = torch.sum((1.0 - probs) * targets, dim=dims)

        tversky = (tp + self.eps) / (
            tp + self.alpha * fp + self.beta * fn + self.eps
        )
        loss = 1.0 - tversky
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Binary focal loss.

    Args:
        gamma: focusing parameter (typically 1 or 2)
        alpha: optional class-balancing weight in [0,1] (None = no alpha)
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B,1,H,W]
            targets: [B,1,H,W] in {0,1}
        """
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t = p if y=1, 1-p if y=0
        p_t = torch.exp(-bce)
        focal_term = (1.0 - p_t) ** self.gamma
        loss = focal_term * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
