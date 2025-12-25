# cnn_aspp/data/ndws_dataset.py
from __future__ import annotations
from pathlib import Path
import glob
import json
import random
import numpy as np

import torch
from torch.utils.data import Dataset


class NdwsSampleTransform:
    """
    Phase 6 augmentation for NDWS tiles.

    Operates on torch tensors:
      - inputs:  FloatTensor [C,H,W]
      - targets: Long/FloatTensor [1,H,W] or [H,W] with {0,1}

    Augmentations:
      - random horizontal/vertical flips
      - random 90° rotations
      - small brightness/contrast jitter on inputs only
    """

    def __init__(self, augment: bool = False):
        self.augment = augment

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor):
        if not self.augment:
            return inputs, targets

        inputs, targets = self._random_flip(inputs, targets)
        inputs, targets = self._random_rot90(inputs, targets)
        inputs = self._brightness_contrast_jitter(inputs)
        return inputs, targets

    def _random_flip(self, x: torch.Tensor, y: torch.Tensor):
        # x: [C,H,W], y: [1,H,W] or [H,W]
        # horizontal flip (W axis)
        if random.random() < 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[-1])
        # vertical flip (H axis)
        if random.random() < 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[-2])
        return x, y

    def _random_rot90(self, x: torch.Tensor, y: torch.Tensor):
        # rotate by k * 90 degrees
        k = random.randint(0, 3)
        if k == 0:
            return x, y
        x = torch.rot90(x, k, dims=[1, 2])
        y = torch.rot90(y, k, dims=[-2, -1])
        return x, y

    def _brightness_contrast_jitter(
        self,
        x: torch.Tensor,
        max_delta: float = 0.1,
        max_scale: float = 0.1,
    ) -> torch.Tensor:
        """
        Apply small brightness/contrast jitter.

        Assumes x is roughly standardized (z-score). This still works:
        it just nudges the mean and scale a bit.
        """
        b = (random.random() * 2.0 - 1.0) * max_delta
        c = 1.0 + (random.random() * 2.0 - 1.0) * max_scale
        x = x * c + b
        return x


class NDWSTilesDataset(Dataset):
    """
    Loads standardized NDWS tiles written by your converter.
    Returns a dict:
      {
        "inputs":  FloatTensor [C,H,W],
        "targets": LongTensor  [1,H,W] with values {0,1},
        "path":    str (debug)
      }

    If stats_path is provided, applies per-channel z-score using stats.json.

    Phase 6:
      - optional data augmentation via `augment` flag.
    """

    def __init__(
        self,
        root: str,
        stats_path: str | None = None,
        eps: float = 1e-6,
        augment: bool = False,   # NEW: Phase 6 aug toggle
    ):
        self.root = Path(root)
        self.files = sorted(glob.glob(str(self.root / "**" / "*.npz"), recursive=True))
        if not self.files:
            raise RuntimeError(f"No NPZ tiles under {root}")
        self.eps = eps

        # stats for z-score
        self.means = None
        self.stds = None
        if stats_path and Path(stats_path).exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)
            chans = stats.get("channels", {})
            # channels are numeric keys "0","1",... in order
            idxs = sorted(int(k) for k in chans.keys())
            self.means = np.array(
                [chans[str(i)]["mean"] for i in idxs],
                dtype=np.float32,
            )[:, None, None]
            self.stds = np.array(
                [chans[str(i)]["std"] for i in idxs],
                dtype=np.float32,
            )[:, None, None]

        # Phase 6: augmentation pipeline
        self.transform = NdwsSampleTransform(augment=augment)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        arr = np.load(p)

        # X: [C,H,W] float32
        x = arr["inputs"].astype(np.float32)

        # Optional z-score normalization
        if self.means is not None and self.stds is not None:
            x = (x - self.means) / (self.stds + self.eps)

        # Y: [1,H,W] int64, ensure {0,1}
        y = arr["targets"].astype(np.int64)
        # sanitize labels: map negatives (e.g., -1) -> 0
        y = np.where(y < 0, 0, y)

        # to torch
        x_t = torch.from_numpy(x)       # FloatTensor [C,H,W]
        y_t = torch.from_numpy(y)       # LongTensor  [1,H,W]

        # Phase 6: apply augmentation (on tensors)
        x_t, y_t = self.transform(x_t, y_t)

        return {
            "inputs": x_t,
            "targets": y_t,
            "path": str(p),
        }
