# cnn_aspp/tasks/train.py
from __future__ import annotations
import os
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning import seed_everything
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore

from cnn_aspp.data.datamodule_ndws import NDWSDataModule

# --- tiny CNN segmentation model ---
class TinySegNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1),
        )

    def forward(self, x):
        return self.net(x)

# --- LightningModule wrapper ---
class LitSeg(pl.LightningModule):
    def __init__(self, in_ch: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TinySegNet(in_ch, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y, _ = batch  # y: [B,1,H,W]
        y = y.float()
        logits = self(x)
        loss = self.loss(logits, y)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            acc = (preds == y.long()).float().mean()

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# --- Default config node we’ll register with Hydra ---
DEFAULT_CFG = OmegaConf.create({
    "seed": 23,
    "dataset": {
        "root": os.environ.get("NDWS_ROOT", "cnn_aspp/data/ndws_out"),
        "stats_path": os.environ.get("NDWS_STATS", "cnn_aspp/data/stats.json"),
        "splits": {
            "train": "${dataset.root}/train",
            "val":   "${dataset.root}/val",
            "test":  "${dataset.root}/test",
        },
        "batch_size": 8,
        "num_workers": 4,
    },
    "model": {
        "in_channels": 12,
        "lr": 1e-3,
    },
    "trainer": {
        "max_epochs": 1,
        "limit_train_batches": 50,
        "limit_val_batches": 10,
        "precision": "32-true",
        "log_every_n_steps": 5,
        "enable_checkpointing": False,
        "enable_model_summary": True,
    },
})

# Register the default config so CLI overrides like trainer.max_epochs=1 are valid pre-main
cs = ConfigStore.instance()
cs.store(name="ndws_default", node=DEFAULT_CFG)

@hydra.main(version_base=None, config_name="ndws_default")
def main(cfg: DictConfig):
    # allow dynamic keys if you want to pass new ones from CLI
    OmegaConf.set_struct(cfg, False)

    print(OmegaConf.to_yaml(cfg, resolve=True))
    seed_everything(cfg.seed, workers=True)

    dm = NDWSDataModule(
        train_dir=cfg.dataset.splits.train,
        val_dir=cfg.dataset.splits.val,
        test_dir=cfg.dataset.splits.test,
        stats_path=cfg.dataset.stats_path,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    )
    model = LitSeg(in_ch=cfg.model.in_channels, lr=cfg.model.lr)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        precision=cfg.trainer.precision,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        enable_model_summary=cfg.trainer.enable_model_summary,
    )
    trainer.fit(model, datamodule=dm)
    print("✅ Smoke training run complete.")

if __name__ == "__main__":
    main()
