import os
import glob
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import onnxruntime
import torch.utils
import torch.utils.data
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from cc.model import Classifier
from cc.dataset import CCDataset

CONFIG: Dict[str, Any] = {
    "epochs": 100,
    "batch_size": 32,
    "num_workers": 8,
    "seed": 0,
    "dataset_path": "data/v3",
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "train_bolt": True,
    "train_hinge": True,
    "use_custom_layers": True,
}


def train() -> L.Trainer:
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    torch.set_float32_matmul_precision("highest")

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
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        drop_last=False,
    )

    trainer = L.Trainer(
        max_epochs=CONFIG["epochs"],
        accelerator="gpu",
        enable_progress_bar=True,
        callbacks=[
            ModelCheckpoint(monitor="val_bolt_cls_acc", mode="max", verbose=True),
            EarlyStopping(monitor="val_bolt_ce_loss", patience=15, verbose=True),
        ],
        log_every_n_steps=20,
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
        val_dataloaders=validation_dataloader,
    )
    print("Classifier training completed.")

    return trainer


def export_onnx_file(model: nn.Module, export_path: str) -> None:
    # Generate a random input
    random_input = torch.randn(1, 1, 224, 224, requires_grad=True)
    # Compute outputs for verification
    with torch.no_grad():
        bolt_config_probs, hinge_config_probs = model(random_input)

    # Export the model
    torch.onnx.export(
        model=model,
        args=random_input,
        f=os.path.join(export_path, "classifier.onnx"),
        export_params=True,
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,
        input_names=["image"],
        output_names=["bolt_config_probs", "hinge_config_probs"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "bolt_config_probs": {0: "batch_size"},
            "hinge_config_probs": {0: "batch_size"},
        },
    )

    ort_session = onnxruntime.InferenceSession(
        path_or_bytes=os.path.join(export_path, "classifier.onnx"),
        providers=["CPUExecutionProvider"],
    )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: random_input.detach().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(bolt_config_probs.numpy(), ort_outputs[0], rtol=0, atol=1e-05)
    np.testing.assert_allclose(hinge_config_probs.numpy(), ort_outputs[1], rtol=0, atol=1e-05)
    print("Exported ONNX model has been tested with ONNXRuntime. Results look good.")


if __name__ == "__main__":
    trainer = train()

    # Load the best checkpoint
    checkpoint_path = glob.glob(f"{trainer.log_dir}/checkpoints/*.ckpt")[0]
    model = Classifier.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location="cpu").eval()
    print(f"Loaded checkpoint from {checkpoint_path}")

    export_onnx_file(model=model, export_path=trainer.log_dir)
