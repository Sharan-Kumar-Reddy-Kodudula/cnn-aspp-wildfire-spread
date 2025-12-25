# ---------------------------
# Hydra/CLI Entrypoint
# File: cnn_aspp/cli/train_aspp_tiny.py
# ---------------------------
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from cnn_aspp.models.aspp_tiny import ASPPTiny
from cnn_aspp.tasks.seg_train import SegLightning, OptimCfg
from cnn_aspp.data.datamodule import SegDataModule
from cnn_aspp.utils.seed import set_seed  # Phase 9: central seeding


@hydra.main(version_base=None, config_path="../conf", config_name="train_aspp_tiny")
def main(cfg: DictConfig):
    # ------------------
    # Phase 9: Reproducibility
    # ------------------
    # Prefer cfg.train.seed; fall back to cfg.seed if present.
    seed = None
    if "train" in cfg and "seed" in cfg.train:
        seed = int(cfg.train.seed)
    elif "seed" in cfg:
        seed = int(cfg.seed)

    set_seed(seed, deterministic=True)

    print("==== Config ====")
    print(OmegaConf.to_yaml(cfg))

    # --- Model ---
    model = ASPPTiny(
        in_channels=cfg.model.in_channels,
        out_channels=1,
        dropout=cfg.model.dropout,              # Phase 6: fuse-block dropout
        focal_gamma=cfg.model.focal_gamma,      # Phase 6: Focal γ
        focal_alpha=cfg.model.focal_alpha,      # optional α (or null)
    )

    task = SegLightning(
        model=model,
        criterion=cfg.train.criterion,          # "tversky" / "bce" / "focal"
        optim_cfg=OptimCfg(**cfg.optim),
        thresh=cfg.train.threshold,
    )

    # --- Data ---
    # Expecting dataset configs like:
    #   dataset:
    #     train:
    #       _target_: cnn_aspp.data.ndws_dataset.NDWSTilesDataset
    #       root: ...
    #       stats_path: ...
    #       augment: true/false
    #     val:
    #       _target_: cnn_aspp.data.ndws_dataset.NDWSTilesDataset
    #       root: ...
    #       stats_path: ...
    #       augment: false
    train_ds = hydra.utils.instantiate(cfg.dataset.train)
    val_ds = hydra.utils.instantiate(cfg.dataset.val)
    dm = SegDataModule(
        train_ds,
        val_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )

    # --- Logging/Callbacks ---
    # NOTE: Hydra changes the working directory to a run dir; we log TB there.
    logger = TensorBoardLogger(save_dir=str(os.getcwd()), name="tb")
    ckpt = ModelCheckpoint(
        monitor="val/IoU",
        mode="max",
        save_top_k=3,
        filename="epoch{epoch:03d}-val_IoU{val/IoU:.3f}",
    )
    lrmon = LearningRateMonitor(logging_interval="step")

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.grad_clip,
        logger=logger,
        callbacks=[ckpt, lrmon],
        log_every_n_steps=cfg.train.log_every_n_steps,
        accumulate_grad_batches=cfg.train.accum,
        devices=cfg.train.devices,
        accelerator=cfg.train.accelerator,
        # Phase 9: determinism flags
        deterministic=True,
        benchmark=False,
    )

    trainer.fit(task, datamodule=dm)
    print("Best ckpt:", ckpt.best_model_path)


if __name__ == "__main__":
    main()
