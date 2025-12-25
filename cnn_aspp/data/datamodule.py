# cnn_aspp/data/datamodule.py
import pytorch_lightning as pl          # <-- add this line
from torch.utils.data import DataLoader

class SegDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, batch_size=8, num_workers=4, pin_memory=True):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
