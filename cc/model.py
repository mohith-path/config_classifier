from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
import torchvision.transforms.v2 as T
from torchvision.models import ResNet18_Weights, resnet18
from torchmetrics.classification import BinaryAccuracy, MulticlassConfusionMatrix


class Classifier(L.LightningModule):

    def __init__(
        self,
        lr: float = 1e-5,
        weight_decay: float = 0,
        train_bolt: bool = True,
        train_hinge: bool = True,
        use_custom_layers: bool = True,
    ) -> None:
        assert train_bolt or train_hinge, "You have to train something buddy"
        super().__init__()

        self._lr = lr
        self._weight_decay = weight_decay
        self.train_bolt = train_bolt
        self.train_hinge = train_hinge
        self.use_custom_layers = use_custom_layers

        # Setup backbone
        self._backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        if self.use_custom_layers:
            self._additional_conv = nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=1, padding="valid"),
                nn.ReLU(),
            )
            feat_dim = 256

        else:
            feat_dim = 512

        # DINO Network
        # self._backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        # feat_dim = self._backbone.embed_dim

        # Freeze backbone
        for param in self._backbone.parameters():
            param.requires_grad = False

        # Set up classified head
        if self.train_bolt:
            self._bolt_prediction_head = nn.Linear(feat_dim, out_features=1)
            self._bolt_prediction_head.weight.data.normal_(mean=0.0, std=0.01)
            self._bolt_prediction_head.bias.data.zero_()

        if self.train_hinge:
            self._hinge_prediction_head = nn.Linear(feat_dim, out_features=1)
            self._hinge_prediction_head.weight.data.normal_(mean=0.0, std=0.01)
            self._hinge_prediction_head.bias.data.zero_()

        # Model preprocessor
        self.pre_processor = T.Compose(
            [
                T.ToDtype(dtype=torch.float32, scale=True),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Setup metrics
        if self.train_bolt:
            self._bolt_train_accuracy = BinaryAccuracy(threshold=0.5, multidim_average="global")
            self._bolt_train_confusion_matrix = MulticlassConfusionMatrix(num_classes=2)

            self._bolt_val_accuracy = BinaryAccuracy(threshold=0.5, multidim_average="global")
            self._bolt_val_confusion_matrix = MulticlassConfusionMatrix(num_classes=2)

        if self.train_hinge:
            self._hinge_train_accuracy = BinaryAccuracy(threshold=0.5, multidim_average="global")
            self._hinge_train_confusion_matrix = MulticlassConfusionMatrix(num_classes=2)

            self._hinge_val_accuracy = BinaryAccuracy(threshold=0.5, multidim_average="global")
            self._hinge_val_confusion_matrix = MulticlassConfusionMatrix(num_classes=2)

    def configure_optimizers(self) -> None:
        optimizer = optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[50],
            gamma=0.25,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self._backbone.conv1(x)
        x = self._backbone.bn1(x)
        x = self._backbone.relu(x)
        x = self._backbone.maxpool(x)

        x = self._backbone.layer1(x)
        x = self._backbone.layer2(x)
        x = self._backbone.layer3(x)
        x = self._backbone.layer4(x)

        if self.use_custom_layers:
            x = self._additional_conv(x)

        x = nn.functional.adaptive_avg_pool2d(input=x, output_size=(1, 1))
        x = torch.flatten(x, 1)

        return x

    def forward(self, x: torch.Tensor, skip_pre_process: bool = False) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        if not skip_pre_process:
            x = self.pre_processor(x)
        x = self._backbone_forward(x)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)

        bolt_logits = self._bolt_prediction_head(x) if self.train_bolt else None
        hinge_logits = self._hinge_prediction_head(x) if self.train_hinge else None

        return bolt_logits, hinge_logits

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        bolt_logits, hinge_logits = self.forward(x)

        loss = 0

        if self.train_bolt:
            bolt_predictions = torch.sigmoid(bolt_logits)
            bolt_loss = nn.functional.binary_cross_entropy(input=bolt_predictions, target=y[:, :1], reduction="mean")
            loss += bolt_loss

            self.log(
                name=f"val_bolt_ce_loss",
                value=bolt_loss,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                logger=True,
                batch_size=len(x),
                add_dataloader_idx=False,
            )

            self._bolt_val_accuracy.update(preds=bolt_predictions, target=y[:, :1])
            self._bolt_val_confusion_matrix.update(preds=bolt_predictions[:, 0].round(), target=y[:, 0])

        if self.train_hinge:
            hinge_predictions = torch.sigmoid(hinge_logits)
            hinge_loss = nn.functional.binary_cross_entropy(input=hinge_predictions, target=y[:, 1:], reduction="mean")
            loss += hinge_loss
            self.log(
                name=f"val_hinge_ce_loss",
                value=hinge_loss,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                logger=True,
                batch_size=len(x),
                add_dataloader_idx=False,
            )

            self._hinge_val_accuracy.update(preds=hinge_predictions, target=y[:, 1:])
            self._hinge_val_confusion_matrix.update(preds=hinge_predictions[:, 0].round(), target=y[:, 1])

        self.logger.experiment.add_image(
            tag=f"val_image_{batch_idx}",
            img_tensor=x[0],
            global_step=self.current_epoch,
        )

        return loss

    def on_validation_epoch_end(self) -> None:
        if self.train_bolt:
            bolt_accuracy = self._bolt_val_accuracy.compute()
            self.log(name=f"val_bolt_cls_acc", value=bolt_accuracy, prog_bar=True, on_epoch=True, on_step=False, logger=True)
            self._bolt_val_accuracy.reset()

            bolt_cm_figure, _ = self._bolt_val_confusion_matrix.plot()
            self.logger.experiment.add_figure(f"val_bolt_confusion_matrix", bolt_cm_figure, global_step=self.current_epoch)
            self._bolt_val_confusion_matrix.reset()

        if self.train_hinge:
            hinge_accuracy = self._hinge_val_accuracy.compute()
            self.log(name=f"val_hinge_cls_acc", value=hinge_accuracy, prog_bar=True, on_epoch=True, on_step=False, logger=True)
            self._hinge_val_accuracy.reset()

            hinge_cm_figure, _ = self._hinge_val_confusion_matrix.plot()
            self.logger.experiment.add_figure(f"val_hinge_confusion_matrix", hinge_cm_figure, global_step=self.current_epoch)
            self._hinge_val_confusion_matrix.reset()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        bolt_logits, hinge_logits = self.forward(x)

        loss = 0

        if self.train_bolt:
            bolt_predictions = torch.sigmoid(bolt_logits)
            bolt_loss = nn.functional.binary_cross_entropy_with_logits(input=bolt_logits, target=y[:, :1], reduction="mean")
            loss += bolt_loss

            self.log(
                name="train_bolt_ce_loss", value=bolt_loss, prog_bar=True, on_epoch=True, on_step=False, logger=True, batch_size=len(x)
            )

            self._bolt_train_accuracy.update(preds=bolt_predictions, target=y[:, :1])
            self._bolt_train_confusion_matrix.update(preds=bolt_predictions[:, 0].round(), target=y[:, 0])

        if self.train_hinge:
            hinge_predictions = torch.sigmoid(hinge_logits)
            hinge_loss = nn.functional.binary_cross_entropy_with_logits(input=hinge_logits, target=y[:, 1:], reduction="mean")
            loss += hinge_loss

            self.log(
                name="train_hinge_ce_loss", value=hinge_loss, prog_bar=True, on_epoch=True, on_step=False, logger=True, batch_size=len(x)
            )

            self._hinge_train_accuracy.update(preds=hinge_predictions, target=y[:, 1:])
            self._hinge_train_confusion_matrix.update(preds=hinge_predictions[:, 0].round(), target=y[:, 1])

        self.logger.experiment.add_image(tag=f"train_image_{batch_idx}", img_tensor=x[0], global_step=self.current_epoch)

        return loss

    def on_train_epoch_end(self) -> None:
        if self.train_bolt:
            bolt_accuracy = self._bolt_train_accuracy.compute()
            self.log(name="train_bolt_cls_acc", value=bolt_accuracy, prog_bar=True, on_epoch=True, on_step=False, logger=True)
            self._bolt_train_accuracy.reset()

            bolt_cm_figure, _ = self._bolt_train_confusion_matrix.plot()
            self.logger.experiment.add_figure("train_bolt_confusion_matrix", bolt_cm_figure, global_step=self.current_epoch)
            self._bolt_train_confusion_matrix.reset()

        if self.train_hinge:
            hinge_accuracy = self._hinge_train_accuracy.compute()
            self.log(name="train_hinge_cls_acc", value=hinge_accuracy, prog_bar=True, on_epoch=True, on_step=False, logger=True)
            self._hinge_train_accuracy.reset()

            hinge_cm_figure, _ = self._hinge_train_confusion_matrix.plot()
            self.logger.experiment.add_figure("train_hinge_confusion_matrix", hinge_cm_figure, global_step=self.current_epoch)
            self._hinge_train_confusion_matrix.reset()

        self.log(name="lr", value=self.optimizers().param_groups[0]["lr"], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.unfreeze_weights()

    def unfreeze_weights(self) -> None:
        # Freeze backbone
        if self.current_epoch == 80:
            print("Unfreezing layer 4...")
            for param in self._backbone.layer4.parameters():
                param.requires_grad = True
