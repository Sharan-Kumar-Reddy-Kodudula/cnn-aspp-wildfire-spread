# cnn_aspp/utils/seed.py
"""
Utilities for deterministic training and reproducibility.

Phase 9: Robustness & Repro
---------------------------

Call set_seed(seed) at the start of every CLI that touches randomness
(train, eval, sweeps, XAI, etc.).
"""

import os
import random
from typing import Optional

import numpy as np
import torch

try:
    # lightning >= 2.0
    import lightning as L  # type: ignore
except ImportError:  # pragma: no cover
    try:
        # legacy name
        import pytorch_lightning as L  # type: ignore
    except ImportError:  # pragma: no cover
        L = None


def set_seed(seed: Optional[int], deterministic: bool = True) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch, and enable deterministic behavior
    where possible.

    Parameters
    ----------
    seed : int or None
        Seed value. If None, this is a no-op.
    deterministic : bool, default True
        If True, enable deterministic flags in PyTorch/cuDNN.
    """
    if seed is None:
        return

    # Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Torch CPU & GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Lightning helper (handles worker_init_fn, etc.)
    if L is not None:
        L.seed_everything(seed, workers=True)

    if deterministic:
        # Make cuDNN as deterministic as possible
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Enforce deterministic algorithms (warn instead of crash if not possible)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # older torch versions may not support warn_only
            torch.use_deterministic_algorithms(True)
