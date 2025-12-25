from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl

from cnn_aspp.models.tiny_cnn import TinyCNN


# ------------ losses ------------
def dice_loss(logits, y, m, eps: float = 1e-6):
    """Soft Dice loss on masked pixels (Bx1xHxW tensors)."""
    p = torch.sigmoid(logits).clamp(eps, 1 - eps)
    p = p * m
    y = y * m
    inter = (p * y).sum()
    denom = p.sum() + y.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def focal_bce_logits(logits, y, m, gamma: float = 2.0, alpha: float = 0.9, eps: float = 1e-6, pos_weight=None):
    """Focal BCE on logits with masking. alpha>0.5 favors positives."""
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none', pos_weight=pos_weight)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')

    p = torch.sigmoid(logits).clamp(eps, 1 - eps)
    pt = p * y + (1 - p) * (1 - y)           # p_t
    pt = pt.clamp(eps, 1 - eps)
    alpha_t = alpha * y + (1 - alpha) * (1 - y)
    focal = alpha_t * (1 - pt).pow(gamma) * bce
    focal = (focal * m).sum() / (m.sum() + eps)
    return focal


# ------------ metrics ------------
@torch.no_grad()
def f1_at_best_thresh(logits, y, m, steps: int = 20):
    """Compute F1 at the best probability threshold over a grid."""
    p = torch.sigmoid(logits)
    yb = (y > 0.5) & (m > 0.5)
    grid = torch.linspace(0.01, 0.99, steps=steps, device=logits.device)
    best = torch.tensor(0.0, device=logits.device)
    for t in grid:
        pred = (p > t) & (m > 0.5)
        tp = (pred & yb).sum().float()
        fp = (pred & (~yb)).sum().float()
        fn = ((~pred) & yb).sum().float()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        best = torch.maximum(best, f1)
    return best


# ------------ dataset ------------
class NPZorPTDataset(Dataset):
    """
    Loads NPZ/PT tiles with keys:
      - inputs:  (C,H,W)
      - targets: (1,H,W), ints {0,1} or any nonzero→positive
      - mask:    (1,H,W) optional; 1=valid, 0=ignore
    Also does:
      - binarize labels, masks
      - per-tile, per-channel z-score normalization
      - optional dilation of positives by DILATE_N pixels (sanity/overfit aid)
      - NaN/Inf sanitization
    """
    def __init__(self, root: Path, pattern: str = '*.npz|*.pt', dilate_px: int = 0):
        self.paths = []
        for ext in pattern.split('|'):
            self.paths += list(Path(root).glob(ext))
        if not self.paths:
            raise FileNotFoundError(f"No tiles under {root} with {pattern}")
        self.dilate_px = int(dilate_px)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p: Path = self.paths[i]
        if p.suffix == '.npz':
            z = np.load(p, allow_pickle=True)
            x = torch.from_numpy(z['inputs']).float()                  # (C,H,W)
            y = torch.from_numpy(z['targets']).float()                 # (1,H,W)
            m = torch.from_numpy(z['mask']) if 'mask' in z else torch.ones_like(y)
        else:
            d = torch.load(p)
            x = torch.as_tensor(d['inputs']).float()
            y = torch.as_tensor(d['targets']).float()
            m = torch.as_tensor(d.get('mask', torch.ones_like(y))).float()

        # binarize
        y = (y > 0).float()
        m = (m > 0.5).float()

        # optional dilation
        if self.dilate_px > 0 and y.any():
            k = 2 * self.dilate_px + 1
            y = F.max_pool2d(y.unsqueeze(0), kernel_size=k, stride=1, padding=self.dilate_px).squeeze(0)
            y = (y > 0).float()

        # per-tile z-norm
        C = x.shape[0]
        flat = x.view(C, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std  = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
        x = (x - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, y, m


def _tiles_with_positives(paths):
    pos = []
    for i, p in enumerate(paths):
        p = Path(p)
        if p.suffix == '.npz':
            z = np.load(p, allow_pickle=True)
            y = torch.from_numpy(z['targets']).float()
            m = torch.from_numpy(z['mask']) if 'mask' in z else torch.ones_like(y)
        else:
            d = torch.load(p)
            y = torch.as_tensor(d['targets']).float()
            m = torch.as_tensor(d.get('mask', torch.ones_like(y))).float()
        if ((y > 0) & (m > 0.5)).any():
            pos.append(i)
    return pos


def make_loaders(root: str, batch_size=1, num_workers=0, pos_only: bool = False,
                 single_tile: bool = False, dilate_px: int = 0):
    train_ds = NPZorPTDataset(Path(root) / 'train', dilate_px=dilate_px)
    val_root = Path(root) / 'val'
    val_ds = NPZorPTDataset(val_root, dilate_px=0) if val_root.exists() else None

    pos_idx = _tiles_with_positives(train_ds.paths)

    # single-tile sanity (repeat one positive tile)
    if single_tile and pos_idx:
        p = train_ds.paths[pos_idx[0]]
        train_ds.paths = [p]
        sampler = WeightedRandomSampler(torch.ones(1), num_samples=20, replacement=True)
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=False)
        val_dl = None if val_ds is None else DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                                        num_workers=num_workers, pin_memory=False)
        return train_dl, val_dl

    # restrict to positive tiles only (optional)
    if pos_only and pos_idx:
        train_ds.paths = [train_ds.paths[i] for i in pos_idx]

    # weighted sampling (oversample positive tiles)
    if pos_only:
        weights = torch.ones(len(train_ds))
    else:
        weights = torch.ones(len(train_ds))
        for i in pos_idx:
            if i < len(weights):
                weights[i] = 10.0
    sampler = WeightedRandomSampler(weights, num_samples=max(len(train_ds) * 4, 64), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=False)
    val_dl = None if val_ds is None else DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                                    num_workers=num_workers, pin_memory=False)
    return train_dl, val_dl


# ------------ Lightning module ------------
class LitSeg(pl.LightningModule):
    def __init__(self, in_ch=12, lr=3e-2, mid: int = 128,
                 alpha: float = 0.9, gamma: float = 2.0,
                 use_dice: bool = True, no_mask: bool = False,
                 pos_weight: float | None = None,
                 pos_prior: float | None = None,
                 dilate: int = 1):
        super().__init__()
        self.save_hyperparameters()
        self.model = TinyCNN(in_ch=in_ch, mid=mid,
                             pos_prior=(pos_prior if pos_prior is not None else 0.02),
                             dilate=dilate)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.use_dice = bool(use_dice)
        self.no_mask = bool(no_mask)
        self.lr = float(lr)
        if pos_weight is not None:
            self.register_buffer("pos_weight_tensor", torch.tensor(float(pos_weight)))
        else:
            self.pos_weight_tensor = None

    def _loss(self, logits, y, m):
        yb = (y > 0.5).float()
        mb = torch.ones_like(m) if self.no_mask else (m > 0.5).float()

        focal = focal_bce_logits(
            logits, yb, mb,
            gamma=self.gamma, alpha=self.alpha,
            pos_weight=self.pos_weight_tensor
        )
        if not self.use_dice:
            return focal
        return focal + dice_loss(logits, yb, mb)

    def step(self, batch, stage='train'):
        x, y, m = batch
        logits = self.model(x)
        loss = self._loss(logits, y, m)
        if torch.isnan(loss) or torch.isinf(loss):
            self.print("NaN/Inf detected; zeroing loss for this step.")
            loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        with torch.no_grad():
            f1_best = f1_at_best_thresh(logits, y, m)

        self.log_dict(
            {f'{stage}_loss_step': loss, f'{stage}_f1_best_step': f1_best},
            prog_bar=True, on_step=True, on_epoch=False
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self.step(batch, 'val')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ------------ main ------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='data/micro')
    ap.add_argument('--in_ch', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--max_steps', type=int, default=200)
    ap.add_argument('--devices', type=int, default=1)
    ap.add_argument('--lr', type=float, default=3e-2)
    ap.add_argument('--mid', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=0)

    # losses / masking
    ap.add_argument('--use_dice', action='store_true')
    ap.add_argument('--no_mask', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.9, help='Focal alpha (pos weight in [0..1])')
    ap.add_argument('--gamma', type=float, default=2.0, help='Focal gamma')

    # class imbalance helpers
    ap.add_argument('--pos_weight', type=float, default=None,
                    help='BCEWithLogits pos_weight (e.g., ~72 for ~1.37%% positives)')
    ap.add_argument('--pos_prior', type=float, default=None,
                    help='Init head bias to logit(pos_prior) (e.g., 0.0137)')

    # sampling / data shaping
    ap.add_argument('--dilate', type=int, default=1, help='Conv dilation in model')
    ap.add_argument('--dilate_px', type=int, default=0, help='Label dilation in dataset')
    ap.add_argument('--pos_only', action='store_true', help='Train only on tiles containing any positives')
    ap.add_argument('--single_tile', action='store_true', help='Repeat a single positive tile (≤20-step overfit)')
    ap.add_argument('--skip_val', action='store_true', help='Disable val loop')

    # PL
    ap.add_argument('--accelerator', type=str, default='auto')
    args = ap.parse_args()

    train_dl, val_dl = make_loaders(
        args.root, args.batch_size, args.num_workers,
        pos_only=args.pos_only, single_tile=args.single_tile, dilate_px=args.dilate_px
    )

    lit = LitSeg(
        in_ch=args.in_ch, lr=args.lr, mid=args.mid,
        alpha=args.alpha, gamma=args.gamma,
        use_dice=args.use_dice, no_mask=args.no_mask,
        pos_weight=args.pos_weight, pos_prior=args.pos_prior,
        dilate=args.dilate
    )

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        log_every_n_steps=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        deterministic=True,
        devices=args.devices,
        accelerator=args.accelerator,
        num_sanity_val_steps=0 if args.skip_val else 2,
        limit_val_batches=0 if args.skip_val else 1,
        gradient_clip_val=1.0,
    )
    trainer.fit(lit, train_dl, None if args.skip_val else val_dl)


if __name__ == '__main__':
    main()
