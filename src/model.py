import torch
from lightning import LightningModule
from monai.losses import (
    DiceCELoss,
    DiceFocalLoss,
    DiceLoss,
    FocalLoss,
    GeneralizedDiceLoss,
)
from monai.metrics import DiceMetric, MeanIoU, compute_dice, compute_iou
from torch.nn import CrossEntropyLoss

from src.models.factory import get_model_factory


class LossFactory:
    def create_loss(self, name):
        if name == "DiceLoss":
            return DiceLoss(sigmoid=True)
        elif name == "DiceFocalLoss":
            return DiceFocalLoss(sigmoid=True)
        elif name == "GeneralizedDiceLoss":
            return GeneralizedDiceLoss(sigmoid=True)
        elif name == "FocalLoss":
            return FocalLoss(sigmoid=True)
        elif name == "DiceCELoss":
            return DiceCELoss(sigmoid=True)
        elif name == "CrossEntropyLoss":
            return CrossEntropyLoss()
        else:
            raise NotImplementedError(f"{name} is not implemented")


class RudrakshaSegModel(LightningModule):
    def __init__(
        self, model_name, smp_encoder, num_classes, loss_fn, lr=1e-4, use_scheduler=True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.model = get_model_factory(
            model_name, num_classes, smp_encoder
        ).create_model()
        self.loss_fn = LossFactory().create_loss(loss_fn)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.iou_metric = MeanIoU(include_background=False, reduction="mean")

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.training_step_outputs.append(loss)
        y_pred = y_pred.sigmoid().round()
        dice = compute_dice(y_pred, y, include_background=False)
        iou = compute_iou(y_pred, y, include_background=False)
        self.log("train_dice", dice.mean())
        self.log("train_iou", iou.mean())
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log("avg_train_loss", avg_loss)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.validation_step_outputs.append(loss)
        y_pred = y_pred.sigmoid().round()
        self.dice_metric(y_pred, y)
        self.iou_metric(y_pred, y)
        return loss

    def on_validation_epoch_end(self):
        self._on_epoch_end("val", self.validation_step_outputs)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("test_loss", loss)
        self.test_step_outputs.append(loss)
        y_pred = y_pred.sigmoid().round()
        self.dice_metric(y_pred, y)
        self.iou_metric(y_pred, y)
        return loss

    def on_test_epoch_end(self):
        self._on_epoch_end("test", self.test_step_outputs)
        self.test_step_outputs.clear()

    def _on_epoch_end(self, mode, step_outputs=None):
        if step_outputs:
            avg_loss = torch.stack(step_outputs).mean()
            self.log(f"avg_{mode}_loss", avg_loss)
            step_outputs.clear()

        dice_metric = self.dice_metric.aggregate().item()
        iou_metric = self.iou_metric.aggregate().item()
        self.log(f"{mode}_dice", dice_metric)
        self.log(f"{mode}_iou", iou_metric)
        self.dice_metric.reset()
        self.iou_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        configuration = {
            "optimizer": optimizer,
            "monitor": "val_loss",  # monitor validation loss
        }

        if self.use_scheduler:
            # Add lr scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",  # or "max" if you're maximizing a metric
                factor=0.5,  # factor by which the learning rate will be reduced
                patience=5,  # number of epochs with no improvement after which learning rate will be reduced
                verbose=True,  # print a message when learning rate is reduced
                threshold=0.001,  # threshold for measuring the new optimum, to only focus on significant changes
            )

            configuration["lr_scheduler"] = scheduler

        return configuration
