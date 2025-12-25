from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from cnn_aspp.models.plain_cnn import PlainCNN

# ---------- Dataset ----------
class NPZorPTDataset(Dataset):
    """
    Loads NPZ/PT tiles with keys:
      - inputs:  (C,H,W)
      - targets: (1,H,W) {0,1}
      - mask:    (1,H,W) optional (1=valid, 0=ignore)

    Binarizes targets/mask, per-tile z-score of inputs, sanitizes NaNs/Infs.
    """
    def __init__(self, root: str, pattern="*.npz|*.pt"):
        root = Path(root)
        paths = []
        for ext in pattern.split("|"):
            paths.extend(root.glob(ext))
        if not paths:
            raise FileNotFoundError(f"No tiles under {root} with pattern {pattern}")
        self.paths = sorted(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        if p.suffix == ".npz":
            z = np.load(p, allow_pickle=True)
            x = torch.from_numpy(z["inputs"]).float()
            y = torch.from_numpy(z["targets"]).float()
            m = torch.from_numpy(z["mask"]) if "mask" in z else torch.ones_like(y)
        else:
            d = torch.load(p)
            x = torch.as_tensor(d["inputs"]).float()
            y = torch.as_tensor(d["targets"]).float()
            m = torch.as_tensor(d.get("mask", torch.ones_like(y)))

        y = (y > 0).float()
        m = (m > 0.5).float()

        C = x.shape[0]
        mean = x.view(C, -1).mean(dim=1, keepdim=True)
        std  = x.view(C, -1).std(dim=1, keepdim=True).clamp_min(1e-6)
        x = (x - mean.view(-1,1,1)) / std.view(-1,1,1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, y, m

# ---------- Loss & Metrics ----------
def tversky_loss(logits, y, m, alpha=0.7, beta=0.3, eps=1e-6):
    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    p = p * m; y = y * m
    tp = (p * y).sum()
    fp = (p * (1.0 - y)).sum()
    fn = ((1.0 - p) * y).sum()
    denom = tp + alpha * fn + beta * fp + eps
    ti = tp / denom
    return 1.0 - ti

@torch.no_grad()
def confusion_terms(logits, y, m, thresh=0.5):
    p = torch.sigmoid(logits)
    pr = (p > thresh) & (m > 0.5)
    gt = (y > 0.5) & (m > 0.5)
    tp = (pr & gt).sum().float()
    tn = ((~pr) & (~gt) & (m > 0.5)).sum().float()
    fp = (pr & (~gt)).sum().float()
    fn = ((~pr) & gt).sum().float()
    return tp, tn, fp, fn

@torch.no_grad()
def prf_oa_at_thresh(logits, y, m, thresh=0.5, eps=1e-9):
    tp, tn, fp, fn = confusion_terms(logits, y, m, thresh=thresh)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    oa        = (tp + tn) / (tp + tn + fp + fn + eps)
    return precision, recall, f1, oa

# ---------- Lightning ----------
class LitPlainSeg(pl.LightningModule):
    def __init__(self, in_ch=12, base_ch=64, lr=4e-4, alpha=0.7, beta=0.3, pos_prior=0.02):
        super().__init__()
        self.save_hyperparameters()
        self.model = PlainCNN(in_ch=in_ch, base_ch=base_ch, pos_prior=pos_prior)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y, m = batch
        logits = self(x)
        loss = tversky_loss(logits, y, m, alpha=self.hparams.alpha, beta=self.hparams.beta)
        prec, rec, f1, oa = prf_oa_at_thresh(logits, y, m, thresh=0.5)

        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_precision": prec,
                f"{stage}_recall": rec,
                f"{stage}_f1": f1,
                f"{stage}_oa": oa,
            },
            prog_bar=(stage != "train"),
            on_epoch=True, on_step=False
        )
        return loss

    def training_step(self, batch, idx):  return self._step(batch, "train")
    def validation_step(self, batch, idx): return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# ---------- Data & main ----------
def make_loaders(root: str, batch_size: int, num_workers: int = 4):
    root = Path(root)
    train_ds = NPZorPTDataset(root / "train")
    val_ds   = NPZorPTDataset(root / "val")
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_dl, val_dl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="cnn_aspp/data/ndws_out", help="Root containing train/ and val/")
    ap.add_argument("--in_ch", type=int, default=12)
    ap.add_argument("--base_ch", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--alpha", type=float, default=0.7, help="Tversky alpha (FN weight)")
    ap.add_argument("--beta",  type=float, default=0.3, help="Tversky beta (FP weight)")
    ap.add_argument("--pos_prior", type=float, default=0.02)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--accelerator", type=str, default="auto")
    args = ap.parse_args()

    train_dl, val_dl = make_loaders(args.root, args.batch_size, args.num_workers)
    lit = LitPlainSeg(in_ch=args.in_ch, base_ch=args.base_ch, lr=args.lr,
                      alpha=args.alpha, beta=args.beta, pos_prior=args.pos_prior)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        deterministic=True,
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=10,
    )
    trainer.fit(lit, train_dl, val_dl)

logger = CSVLogger("lightning_logs", name="plain_cnn")

trainer = Trainer(
    max_epochs=args.max_epochs,
    accelerator=args.accelerator,
    devices=args.devices,
    log_every_n_steps=10,
    logger=logger,
)

if __name__ == "__main__":
    main()
