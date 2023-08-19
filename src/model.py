import torch
from lightning import LightningModule
from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet


class RudrakshaSegModel(LightningModule):
    def __init__(self, num_classes, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.model = UNet(
            in_channels=3,
            out_channels=num_classes,
            spatial_dims=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        # self.loss_fn = DiceLoss(sigmoid=True)
        self.loss_fn = GeneralizedDiceLoss(sigmoid=True)
        self.metric = DiceMetric(include_background=False, reduction="mean")

        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.metric(y_pred, y)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        dice_metric = self.metric.aggregate().item()
        self.log("avg_val_loss", avg_val_loss)
        self.log("val_dice", dice_metric)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("test_loss", loss)
        self.metric(y_pred, y)
        self.test_step_outputs.append(loss)
        return loss

    def on_test_epoch_end(self):
        avg_test_loss = torch.stack(self.test_step_outputs).mean()
        test_dice_metric = self.metric.aggregate().item()
        self.log("avg_test_loss", avg_test_loss)
        self.log("test_dice", test_dice_metric)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Add lr scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # or "max" if you're maximizing a metric
            factor=0.5,  # factor by which the learning rate will be reduced
            patience=5,  # number of epochs with no improvement after which learning rate will be reduced
            verbose=True,  # print a message when learning rate is reduced
            threshold=0.001,  # threshold for measuring the new optimum, to only focus on significant changes
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",  # monitor validation loss
        }
