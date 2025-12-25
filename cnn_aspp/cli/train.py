# cnn_aspp/cli/train.py
from __future__ import annotations

import inspect
import os

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from cnn_aspp.models.aspp_tiny import ASPPTiny
from cnn_aspp.tasks.seg_task import SegTrainTask
from cnn_aspp.data.datamodule_ndws import NDWSDataModule
from cnn_aspp.utils.seed import set_seed


# ---------- Model adapter ----------
def _build_aspp_tiny(cfg_model) -> ASPPTiny:
    sig = inspect.signature(ASPPTiny.__init__)
    params = sig.parameters
    kw = {}

    # in_channels / in_ch
    in_ch_val = getattr(cfg_model, "in_channels", 12)
    if "in_channels" in params:
        kw["in_channels"] = in_ch_val
    elif "in_ch" in params:
        kw["in_ch"] = in_ch_val

    # filters / width / base_channels ...
    width_val = getattr(cfg_model, "filters", 32)
    for alias in ("filters", "width", "base_channels", "base_width", "features", "planes", "channels"):
        if alias in params:
            kw[alias] = width_val
            break

    # dilations / rates
    rates_val = list(getattr(cfg_model, "dilations", [1, 3, 6, 12]))
    for alias in ("dilations", "rates", "atrous_rates", "aspp_rates"):
        if alias in params:
            kw[alias] = rates_val
            break

    # out channels / classes (binary)
    for alias in ("out_channels", "out_ch", "num_classes"):
        if alias in params:
            kw[alias] = 1
            break

    kw = {k: v for k, v in kw.items() if k in params}
    return ASPPTiny(**kw)


# ---------- DataModule adapter ----------
def _build_ndws_datamodule(cfg_task, cfg_dataset):
    """
    Build NDWSDataModule from generic dataset/task config.

    IMPORTANT: we deliberately do NOT read cfg_dataset.train_dir / val_dir / test_dir
    because they may contain broken interpolations like ${root}/train/EVTUNK.
    Instead, we derive directory paths from dataset.root only.
    """
    sig = inspect.signature(NDWSDataModule.__init__)
    params = sig.parameters

    def first_supported(pairs):
        for k, v in pairs:
            if k in params and v is not None:
                return k, v
        return None, None

    # Required: dataset.root
    root = cfg_dataset.root
    # Optional: dataset.split (micro / ndws / etc.), passed through if the DM supports it
    split = getattr(cfg_dataset, "split", "train")

    # Derive dirs purely from root
    # For NDWS, typical layout is: <root>/train, <root>/val, <root>/test
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")

    dm_kwargs = {}

    # root/split if supported
    k, v = first_supported([
        ("root", root), ("data_root", root), ("data_dir", root),
        ("dataset_root", root), ("ndws_root", root), ("path", root),
    ])
    if k:
        dm_kwargs[k] = v
    k, v = first_supported([
        ("split", split), ("subset", split), ("which", split),
        ("partition", split), ("stage", split),
    ])
    if k:
        dm_kwargs[k] = v

    # explicit dirs if supported
    k, v = first_supported([("train_dir", train_dir), ("train_root", train_dir), ("train_path", train_dir)])
    if k:
        dm_kwargs[k] = v
    k, v = first_supported([("val_dir", val_dir), ("valid_dir", val_dir), ("val_root", val_dir), ("val_path", val_dir)])
    if k:
        dm_kwargs[k] = v
    k, v = first_supported([("test_dir", test_dir), ("test_root", test_dir), ("test_path", test_dir)])
    if k:
        dm_kwargs[k] = v

    # batch/workers
    k, v = first_supported([("batch_size", cfg_task.batch_size), ("bs", cfg_task.batch_size)])
    if k:
        dm_kwargs[k] = v
    k, v = first_supported([("num_workers", cfg_task.num_workers), ("workers", cfg_task.num_workers)])
    if k:
        dm_kwargs[k] = v

    # normalize + augs
    k, v = first_supported([("normalize", cfg_dataset.normalize), ("do_normalize", cfg_dataset.normalize)])
    if k:
        dm_kwargs[k] = v
    k, v = first_supported([
        ("aug_flip", cfg_dataset.augment.flip), ("flip", cfg_dataset.augment.flip),
        ("hflip", cfg_dataset.augment.flip), ("random_flip", cfg_dataset.augment.flip),
    ])
    if k:
        dm_kwargs[k] = v
    k, v = first_supported([
        ("aug_rot", cfg_dataset.augment.rotate), ("rotate", cfg_dataset.augment.rotate),
        ("rot", cfg_dataset.augment.rotate), ("random_rotate", cfg_dataset.augment.rotate),
    ])
    if k:
        dm_kwargs[k] = v

    # optional extras
    if hasattr(cfg_dataset, "channels") and cfg_dataset.channels is not None:
        k, v = first_supported([
            ("channels", cfg_dataset.channels), ("channel_names", cfg_dataset.channels),
            ("selected_channels", cfg_dataset.channels),
        ])
        if k:
            dm_kwargs[k] = v
    if hasattr(cfg_dataset, "stats_path"):
        k, v = first_supported([("stats_path", cfg_dataset.stats_path), ("stats", cfg_dataset.stats_path)])
        if k:
            dm_kwargs[k] = v

    for opt, val, aliases in [
        ("tile_size", getattr(cfg_dataset, "tile_size", None), ("tile_size", "tile", "patch_size")),
        ("dtype", getattr(cfg_dataset, "dtype", None), ("dtype",)),
        ("target_dtype", getattr(cfg_dataset, "target_dtype", None), ("target_dtype", "mask_dtype")),
        ("normalization", getattr(cfg_dataset, "normalization", None), ("normalization", "norm_cfg")),
    ]:
        if val is not None:
            for alias in aliases:
                if alias in params:
                    dm_kwargs[alias] = val
                    break

    # Build the user's DM with supported kwargs
    inner = NDWSDataModule(**{k: v for k, v in dm_kwargs.items() if k in params})

    # Force-override attributes so setup() uses our dirs and stats
    for attr, value in (("train_dir", train_dir), ("val_dir", val_dir), ("test_dir", test_dir)):
        if hasattr(inner, attr) and value is not None:
            setattr(inner, attr, value)
    if hasattr(inner, "stats_path") and getattr(cfg_dataset, "stats_path", None):
        inner.stats_path = cfg_dataset.stats_path

    # If it's already a LightningDataModule, return; else wrap
    if isinstance(inner, pl.LightningDataModule):
        return inner

    class _Adapter(pl.LightningDataModule):
        def __init__(self, obj):
            super().__init__()
            self.obj = obj

        def prepare_data(self):
            return getattr(self.obj, "prepare_data", lambda: None)()

        def setup(self, stage=None):
            return getattr(self.obj, "setup", lambda s=None: None)(stage)

        def train_dataloader(self):
            return self.obj.train_dataloader()

        def val_dataloader(self):
            return getattr(self.obj, "val_dataloader", lambda: [])()

        def test_dataloader(self):
            return getattr(self.obj, "test_dataloader", lambda: [])()

    return _Adapter(inner)


@hydra.main(config_path="../conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # Seed handling
    seed = None
    if "seed" in cfg:
        seed = cfg.seed
    elif "task" in cfg and "seed" in cfg.task:
        seed = cfg.task.seed
    elif "trainer" in cfg and "seed" in cfg.trainer:
        seed = cfg.trainer.seed
    if seed is not None:
        set_seed(int(seed), deterministic=True)

    # Print config (no resolve=True to avoid broken ${root})
    print(OmegaConf.to_yaml(cfg))

    # Model
    model = _build_aspp_tiny(cfg.model)

    # Training task
    task = SegTrainTask(
        model=model,
        threshold=cfg.task.threshold,
        alpha=cfg.task.tversky.alpha,
        beta=cfg.task.tversky.beta,
        optim=dict(cfg.task.optim),
        sched=dict(cfg.task.scheduler),
    )

    # Data
    dm = _build_ndws_datamodule(cfg.task, cfg.dataset)

    # Logging config with defaults
    tb_dir = getattr(cfg.logging, "tb_dir", "tb")
    csv_dir = getattr(cfg.logging, "csv_dir", "lightning_logs")
    version = getattr(cfg.logging, "version", None)
    log_every = getattr(cfg.logging, "log_every_n_steps", 50)

    tb_logger = TensorBoardLogger(save_dir=tb_dir, name="", version=version)
    csv_logger = CSVLogger(save_dir=csv_dir, name="", version=version)

    # Callbacks
    ckpt = ModelCheckpoint(
        monitor="val/iou",
        mode="max",
        save_top_k=1,
        filename="epoch={epoch}-val_iou={val/iou:.3f}",
        auto_insert_metric_name=False,
        save_last=False,
    )
    lrmon = LearningRateMonitor(logging_interval="epoch")

    # Trainer kwargs
    amp_ok = bool(getattr(cfg.task, "amp", False)) and torch.cuda.is_available()
    precision = "16-mixed" if amp_ok else "32-true"

    fast_dev_run = getattr(cfg.trainer, "fast_dev_run", False)
    persistent_workers = getattr(cfg.trainer, "persistent_workers", False)

    requested_kwargs = dict(
        max_epochs=cfg.task.epochs,
        deterministic=True,
        logger=[tb_logger, csv_logger],
        devices="auto",
        accelerator="auto",
        precision=precision,
        log_every_n_steps=log_every,
        enable_checkpointing=True,
        callbacks=[ckpt, lrmon],
        fast_dev_run=fast_dev_run,
        persistent_workers=persistent_workers,
    )

    # Only keep kwargs supported by this PL version
    trainer_params = inspect.signature(pl.Trainer.__init__).parameters
    trainer_kwargs = {k: v for k, v in requested_kwargs.items() if k in trainer_params}

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(task, datamodule=dm)


if __name__ == "__main__":
    main()
