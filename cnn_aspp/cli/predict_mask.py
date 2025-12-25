# ---------------------------
# Inference Script
# File: cnn_aspp/cli/predict_mask.py
# ---------------------------
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from cnn_aspp.models.aspp_tiny import ASPPTiny


def load_tensor(path: Path) -> torch.Tensor:
    """
    Load a single NDWS-style tile from .npz or .npy and return a tensor [1,C,H,W].
    """
    arr = np.load(path)

    # Handle .npz (NpzFile) and .npy (ndarray)
    if isinstance(arr, np.lib.npyio.NpzFile):
        # Try common keys used in our project
        for key in ("x", "inputs", "data", "arr_0"):
            if key in arr:
                arr = arr[key]
                break
        else:
            raise KeyError(
                f"No known array key found in NPZ file {path}. "
                f"Available keys: {list(arr.keys())}"
            )

    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(arr)} from {path}")

    t = torch.from_numpy(arr).float()

    # Expect [C,H,W] or [1,C,H,W]; ensure [1,C,H,W]
    if t.ndim == 3:
        t = t.unsqueeze(0)
    elif t.ndim == 2:
        # [H,W] → [1,1,H,W] (unlikely for our case, but safe)
        t = t.unsqueeze(0).unsqueeze(0)
    return t


def main():
    ap = argparse.ArgumentParser(description="Predict fire mask for a single NDWS tile.")
    ap.add_argument("checkpoint", help="Path to ASPPTiny checkpoint (.ckpt)")
    ap.add_argument("input_npy", help="Path to input .npz/.npy tile")
    ap.add_argument(
        "--in_channels",
        type=int,
        default=8,
        help="Number of input channels (use 12 for NDWS ASPPTiny)",
    )
    ap.add_argument(
        "--out_png",
        type=str,
        default="pred.png",
        help="Where to save the predicted binary mask PNG",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for converting logits to binary mask",
    )
    args = ap.parse_args()

    # Build model with correct input channels
    model = ASPPTiny(in_channels=args.in_channels)
    state = torch.load(args.checkpoint, map_location="cpu")

    # Support PL checkpoint (contains 'state_dict') or raw state_dict
    sd = state.get("state_dict", state)
    filtered_sd = {
        k.replace("model.", ""): v
        for k, v in sd.items()
        if k.startswith("model.") or k in model.state_dict()
    }
    model.load_state_dict(filtered_sd)
    model.eval()

    # Load input tile
    x = load_tensor(Path(args.input_npy))  # [1,C,H,W]

    with torch.no_grad():
        p = model.predict(x)  # [1,1,H,W] probas or logits depending on implementation
        pred = (p >= args.threshold).float()

    save_image(pred, args.out_png)
    print("Saved:", args.out_png)


if __name__ == "__main__":
    main()
 