from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
import torchvision.transforms.v2 as T
from torchvision.models import ResNet50_Weights, resnet50
from torchmetrics.classification import BinaryAccuracy, MulticlassConfusionMatrix


class Classifier(L.LightningModule):

    def __init__(self, lr: float = 1e-5) -> None:

        super().__init__()

        self._lr = lr

        # Setup backbone
        self._backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = self._backbone.fc.in_features
        self._backbone.fc = nn.Identity()

        # self._backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        # feat_dim = self._backbone.embed_dim

        # Freeze backbone
        for param in self._backbone.parameters():
            param.requires_grad = False

        # Set up classified head
        self._prediction_head = nn.Linear(feat_dim, out_features=1)
        self._prediction_head.weight.data.normal_(mean=0.0, std=0.01)
        self._prediction_head.bias.data.zero_()

        # Model preprocessor
        self.pre_processor = T.Compose(
            [
                T.ToDtype(dtype=torch.float32, scale=True),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Setup metrics
        self._val_accuracy = BinaryAccuracy(threshold=0.5, multidim_average="global")
        self._train_accuracy = BinaryAccuracy(threshold=0.5, multidim_average="global")

        self._val_confusion_matrix = MulticlassConfusionMatrix(num_classes=2)
        self._train_confusion_matrix = MulticlassConfusionMatrix(num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_processor(x)
        x = self._backbone(x)
        x = self._prediction_head(x)

        return x

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)
        predictions = torch.sigmoid(logits)

        loss = nn.functional.binary_cross_entropy(input=predictions, target=y, reduction="mean")
        self.log(name="val_bolt_ce_loss", value=loss, prog_bar=True, on_epoch=True, on_step=False, logger=True, batch_size=len(x))

        self._val_accuracy.update(preds=predictions, target=y)
        self._val_confusion_matrix.update(preds=predictions[:, 0].round(), target=y[:, 0])

        return loss

    def on_validation_epoch_end(self) -> None:
        accuracy = self._val_accuracy.compute()
        self.log(name="val_bolt_cls_acc", value=accuracy, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        self._val_accuracy.reset()

        bolt_cm_figure, _ = self._val_confusion_matrix.plot()
        self.logger.experiment.add_figure("val_bolt_confusion_matrix", bolt_cm_figure, global_step=self.current_epoch)
        self._val_confusion_matrix.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)
        predictions = torch.sigmoid(logits)
        loss = nn.functional.binary_cross_entropy_with_logits(input=logits, target=y, reduction="mean")

        self.log(name="train_bolt_ce_loss", value=loss, prog_bar=True, on_epoch=True, on_step=False, logger=True, batch_size=len(x))

        self._train_accuracy.update(preds=predictions, target=y)
        self._train_confusion_matrix.update(preds=predictions[:, 0].round(), target=y[:, 0])

        return loss

    def on_train_epoch_end(self) -> None:
        accuracy = self._train_accuracy.compute()
        self.log(name="train_bolt_cls_acc", value=accuracy, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        self._train_accuracy.reset()

        bolt_cm_figure, _ = self._train_confusion_matrix.plot()
        self.logger.experiment.add_figure("train_bolt_confusion_matrix", bolt_cm_figure, global_step=self.current_epoch)
        self._train_confusion_matrix.reset()

    def configure_optimizers(self) -> None:
        optimizer = optim.Adam(self.parameters(), lr=self._lr)

        return optimizer
