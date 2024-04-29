from typing import Any, Dict

import numpy as np
import torch
import torch.utils
import torch.utils.data
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from cc.model import Classifier
from cc.dataset import CCDataset

CONFIG: Dict[str, Any] = {
    "epochs": 200,
    "batch_size": 32,
    "num_workers": 8,
    "seed": 0,
    "dataset_path": "data/v2",
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "train_bolt": True,
    "train_hinge": True,
    "use_custom_layers": True,
}


def train():
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    train_dataset = CCDataset(path=CONFIG["dataset_path"], type="train")
    validation_dataset = CCDataset(path=CONFIG["dataset_path"], type="val")

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        drop_last=True,
    )

    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        drop_last=False,
    )

    validation_dataset_2 = CCDataset(path="data/v2-1", type="val")
    validation_dataloader_2 = torch.utils.data.DataLoader(
        dataset=validation_dataset_2,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        drop_last=False,
    )

    trainer = L.Trainer(
        max_epochs=CONFIG["epochs"],
        accelerator="gpu",
        enable_progress_bar=True,
        callbacks=[
            ModelCheckpoint(monitor="val_bolt_cls_acc", mode="max", verbose=True),
            # EarlyStopping(monitor="val_bolt_ce_loss", patience=15, verbose=True),
        ],
    )

    model = Classifier(
        lr=CONFIG.get("lr"),
        weight_decay=CONFIG.get("weight_decay"),
        train_bolt=CONFIG.get("train_bolt"),
        train_hinge=CONFIG.get("train_hinge"),
        use_custom_layers=CONFIG.get("use_custom_layers"),
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=[validation_dataloader, validation_dataloader_2],
    )


if __name__ == "__main__":
    train()
