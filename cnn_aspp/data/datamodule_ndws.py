# cnn_aspp/data/datamodule_ndws.py
from __future__ import annotations

from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule  # modern import

from .ndws_dataset import NDWSTilesDataset


class NDWSDataModule(LightningDataModule):
    def __init__(
        self,
        train_dir,
        val_dir,
        test_dir=None,
        stats_path=None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_augment: bool = False,  # Phase 6: aug flags per split
        val_augment: bool = False,
        test_augment: bool = False,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.stats_path = stats_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_augment = train_augment
        self.val_augment = val_augment
        self.test_augment = test_augment

        # placeholders set in setup()
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

    # (prepare_data not overridden; Lightning will skip the override check)
    def setup(self, stage=None):
        # Training / validation
        if stage in (None, "fit"):
            self.ds_train = NDWSTilesDataset(
                root=self.train_dir,
                stats_path=self.stats_path,
                augment=self.train_augment,  # Phase 6: train aug toggle
            )
            self.ds_val = NDWSTilesDataset(
                root=self.val_dir,
                stats_path=self.stats_path,
                augment=self.val_augment,    # usually False
            )

        # Test
        if stage in (None, "test") and self.test_dir:
            self.ds_test = NDWSTilesDataset(
                root=self.test_dir,
                stats_path=self.stats_path,
                augment=self.test_augment,   # usually False
            )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        if self.ds_test is None:
            return None
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
