from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 (may be useful later)
from typing import Iterable, Optional, Tuple, Dict


# ---------------------------
# Building blocks
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None, d: int = 1):
        super().__init__()
        if p is None:
            p = (k // 2) * d
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # Optionally keep track of dilation on the conv (handy for debugging)
        self.dilation = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling with 4 branches, dilations {1, 3, 6, 12}.
    Each branch: 3x3 conv -> 32 channels, BN, ReLU.
    Concatenate along channel dim.
    """

    def __init__(self, in_ch: int, branch_out: int = 32, dilations: Iterable[int] = (1, 3, 6, 12)):
        super().__init__()
        # Keep track of dilations so we can align them with branches for XAI
        self.dilations = list(dilations)

        self.branches = nn.ModuleList([
            ConvBNReLU(in_ch, branch_out, k=3, d=d)
            for d in dilations
        ])
        self.out_ch = branch_out * len(tuple(dilations))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        return torch.cat(feats, dim=1)


# ---------------------------
# Losses
# ---------------------------
class TverskyLoss(nn.Module):
    """Tversky loss for binary segmentation.

    L = 1 - TverskyIndex, where
        TI = TP / (TP + alpha*FP + beta*FN)
    Uses probabilities (sigmoid(logits)).
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-7):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # ensure same shape [N,1,H,W]
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            probs = probs * mask
            targets = targets * mask

        tp = (probs * targets).sum(dim=(0, 2, 3))
        fp = (probs * (1.0 - targets)).sum(dim=(0, 2, 3))
        fn = ((1.0 - probs) * targets).sum(dim=(0, 2, 3))

        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = 1.0 - ti
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Binary focal loss for segmentation, with optional mask.

    Args:
        gamma: focusing parameter (typically 1 or 2)
        alpha: optional class-balancing weight in [0,1] (None = no alpha)
        eps: small value for numerical stability
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None, eps: float = 1e-7):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [N,1,H,W]
            targets: [N,1,H,W] or [N,H,W] in {0,1}
            mask: optional [N,1,H,W] or [N,H,W] binary mask
        """
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # per-pixel BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if mask is not None:
            bce = bce * mask

        # p_t = p if y=1 else 1-p
        p_t = torch.exp(-bce)
        focal_term = (1.0 - p_t) ** self.gamma
        loss = focal_term * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            if mask is not None:
                alpha_t = alpha_t * mask
            loss = alpha_t * loss

        # avoid NaNs if everything is masked out
        denom = (loss.numel() if mask is None else mask.sum() + self.eps)
        return loss.sum() / denom


# ---------------------------
# Model
# ---------------------------
class ASPPTiny(nn.Module):
    """
    Phase 4–6 model:
      Stem: Conv->64 -> Conv->128 (3x3, BN, ReLU)
      ASPP: 4 branches with dilations {1,3,6,12}, each 3x3->32
      Fuse: 3x3->32 -> (Dropout p=0.1) -> 3x3->32 (+BN) -> 1x1->1 (logits)
      predict() = sigmoid(logits)

    Phase 6 extras:
      - Dropout (p configurable) in fuse block.
      - compute_loss supports 'tversky', 'bce', and 'focal'.

    Phase 8 hooks:
      - self.aspp_branches: per-dilation ASPP conv branches (for Grad-CAM)
      - self.final_conv: last conv layer before sigmoid (for Grad-CAM)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        dropout: float = 0.1,         # Phase 6: fuse-block dropout
        focal_gamma: float = 2.0,     # Phase 6: Focal loss γ
        focal_alpha: float | None = None,  # optional focal α
    ):
        super().__init__()
        # Stem
        self.stem1 = ConvBNReLU(in_channels, 64, k=3)
        self.stem2 = ConvBNReLU(64, 128, k=3)

        # ASPP
        self.aspp = ASPP(128, branch_out=32, dilations=(1, 3, 6, 12))  # -> 32*4 = 128 ch

        # Phase 8: expose per-dilation ASPP branches as a ModuleDict
        # Map dilations [1,3,6,12] to keys "d1","d3","d6","d12"
        self.aspp_branches = nn.ModuleDict({
            f"d{d}": branch
            for d, branch in zip(self.aspp.dilations, self.aspp.branches)
        })

        # Fuse
        self.fuse1 = ConvBNReLU(128, 32, k=3)
        self.fuse2 = ConvBNReLU(32, 32, k=3)
        # Phase 6: dropout in fuse block
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

        # Head (final conv that produces logits)
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

        # Phase 8: final conv handle for Grad-CAM
        self.final_conv = self.head

        # Losses
        self.tversky = TverskyLoss(alpha=0.7, beta=0.3)
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    # --- core API ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.aspp(x)
        x = self.fuse1(x)
        x = self.dropout(x)      # Phase 6: regularization
        x = self.fuse2(x)
        logits = self.head(x)
        return logits  # [N,1,H,W]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        criterion: str = "tversky",
        focal_gamma: Optional[float] = None,   # optional override per call
        focal_alpha: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss; robust to mask being a tuple/list from some datasets.

        criterion ∈ {"tversky", "bce", "focal"}.
        """
        # sanitize mask: allow (mask, meta) or empty tuple/list
        if isinstance(mask, (list, tuple)):
            mask = mask[0] if (len(mask) > 0 and torch.is_tensor(mask[0])) else None

        criterion = criterion.lower()

        if criterion == "tversky":
            loss = self.tversky(logits, targets, mask)

        elif criterion in ("bce", "bce_logits", "bcewithlogits"):
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
            targets = targets.float()
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                loss = self.bce(logits * mask, targets * mask)
            else:
                loss = self.bce(logits, targets)

        elif criterion in ("focal", "focal_loss"):
            # optionally override gamma/alpha per-call if provided
            gamma = focal_gamma if focal_gamma is not None else self.focal.gamma
            alpha = focal_alpha if focal_alpha is not None else self.focal.alpha
            focal = FocalLoss(gamma=gamma, alpha=alpha)
            loss = focal(logits, targets, mask)

        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        return {"loss": loss}


# ---------------------------
# Quick self-test (DoD)
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    for C in (1, 3, 8):
        model = ASPPTiny(in_channels=C)
        x = torch.randn(2, C, 128, 128)
        logits = model(x)
        assert logits.shape == (2, 1, 128, 128), f"Bad logits shape: {logits.shape}"
        with torch.no_grad():
            p = model.predict(x)
            assert p.min().ge(0).item() and p.max().le(1).item(), "predict() not in [0,1]"

        # Phase 8: check Grad-CAM plumbing
        assert isinstance(model.aspp_branches, nn.ModuleDict)
        assert set(model.aspp_branches.keys()) == {"d1", "d3", "d6", "d12"}
        assert isinstance(model.final_conv, nn.Conv2d)

    print("DoD passed: forward [2,C,128,128] -> [2,1,128,128], predict() in [0,1], XAI hooks OK.")
