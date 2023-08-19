import torch
from lightning import LightningModule
from monai.losses import DiceLoss
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
        self.loss_fn = DiceLoss(sigmoid=True)
        self.metric = DiceMetric(include_background=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        dice_score = self.metric(y_hat, y)
        self.log_dict(
            {"train_loss": loss, "train_dice": dice_score.mean()},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        dice_score = self.metric(y_pred, y)
        self.log_dict(
            {"val_loss": loss, "val_dice": dice_score.mean()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return dice_score

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
