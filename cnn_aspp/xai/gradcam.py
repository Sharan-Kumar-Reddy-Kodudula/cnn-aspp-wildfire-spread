# cnn_aspp/xai/gradcam.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GradCAMOutput:
    cam: torch.Tensor          # [B, 1, H, W], normalized to [0, 1]
    probs: torch.Tensor        # [B, 1, H, W], sigmoid(logits)
    logits: torch.Tensor       # [B, 1, H, W], raw model output


class GradCAM:
    """
    Simple Grad-CAM helper bound to a single target layer.

    Assumptions:
      - model(x) returns logits of shape [B,1,H,W].
      - target_layer is part of model.forward.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device or next(model.parameters()).device

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def _compute_target_scalar(
        self,
        probs: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert [B,1,H,W] prob maps into scalar per sample for backprop.
        """
        if target_mask is None:
            return probs.mean(dim=(1, 2, 3))  # [B]
        else:
            assert probs.shape == target_mask.shape, (
                f"Shape mismatch probs {probs.shape}, mask {target_mask.shape}"
            )
            num = (probs * target_mask).flatten(1).sum(dim=1)
            den = target_mask.flatten(1).sum(dim=1).clamp_min(1e-6)
            return num / den

    def __call__(
        self,
        x: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> GradCAMOutput:
        """
        Compute Grad-CAM for a batch.
        """
        self.model.eval()
        x = x.to(self.device)
        if target_mask is not None:
            target_mask = target_mask.to(self.device)

        self.model.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(True):
            logits = self.model(x)  # [B,1,H,W]
            probs = torch.sigmoid(logits)
            target_scalar = self._compute_target_scalar(probs, target_mask)
            target_scalar.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "Check that target_layer is used in model.forward()."
            )

        A = self.activations  # [B,C,H,W]
        G = self.gradients    # [B,C,H,W]

        weights = G.mean(dim=(2, 3), keepdim=True)      # [B,C,1,1]
        cam = (weights * A).sum(dim=1, keepdim=True)    # [B,1,H,W]
        cam = F.relu(cam)

        B = cam.size(0)
        cam_flat = cam.view(B, -1)
        max_vals, _ = cam_flat.max(dim=1, keepdim=True)
        cam_flat = cam_flat / (max_vals + 1e-6)
        cam = cam_flat.view_as(cam)

        return GradCAMOutput(
            cam=cam.detach(),
            probs=probs.detach(),
            logits=logits.detach(),
        )


def upsample_to_match(
    src: torch.Tensor,
    ref: torch.Tensor,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Upsample src to match ref's H,W.
    """
    _, _, H, W = ref.shape
    return F.interpolate(src, size=(H, W), mode=mode, align_corners=align_corners)
