import segmentation_models_pytorch as smp
from monai.networks.nets import UNETR, SwinUNETR, UNet

from src.models.attention_res_unet import AttResUNet
from src.models.attention_unet import AttentionUNet
from src.models.fcn import FCN8s
from src.models.res_unet import ResUNet
from src.models.unet import UNet as CustomUNet


class BaseModelFactory:
    def __init__(self, num_classes, smp_encoder="resnet18"):
        self.num_classes = num_classes
        self.smp_encoder = smp_encoder

    def create_model(self, name):
        raise NotImplementedError


class CustomFCN8sFactory(BaseModelFactory):
    def create_model(self):
        return FCN8s(num_classes=self.num_classes)


class CustomUNetFactory(BaseModelFactory):
    def create_model(self):
        return CustomUNet(
            num_classes=self.num_classes,
        )


class CustomResUNetFactory(BaseModelFactory):
    def create_model(self):
        return ResUNet(
            num_classes=self.num_classes,
        )


class AttResUNetFactory(BaseModelFactory):
    def create_model(self):
        return AttResUNet(
            num_classes=self.num_classes,
        )


class AttentionUNetFactory(BaseModelFactory):
    def create_model(self):
        return AttentionUNet(
            num_classes=self.num_classes,
        )


class UNetFactory(BaseModelFactory):
    def create_model(self):
        return UNet(
            in_channels=3,
            out_channels=self.num_classes,
            spatial_dims=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )


class DeepLabV3Factory(BaseModelFactory):
    def create_model(self):
        return smp.DeepLabV3(
            encoder_name=self.smp_encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes,
        )


class DeepLabV3PlusFactory(BaseModelFactory):
    def create_model(self):
        return smp.DeepLabV3Plus(
            encoder_name=self.smp_encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes,
        )


class FPNFactory(BaseModelFactory):
    def create_model(self):
        return smp.FPN(
            encoder_name=self.smp_encoder,
            encoder_weights="imagenet",
            classes=self.num_classes,
        )


class SmpResUNetFactory(BaseModelFactory):
    def create_model(self):
        return smp.Unet(
            encoder_name=self.smp_encoder,
            encoder_weights="imagenet",
            classes=self.num_classes,
        )


class SmpResUNetPlusPlusFactory(BaseModelFactory):
    def create_model(self):
        return smp.UnetPlusPlus(
            encoder_name=self.smp_encoder,
            encoder_weights="imagenet",
            classes=self.num_classes,
        )


class UNETRFactory(BaseModelFactory):
    def create_model(self):
        return UNETR(
            in_channels=3,
            out_channels=self.num_classes,
            img_size=256,
            norm_name="batch",
            spatial_dims=2,
        )


class SwinUNETRFactory(BaseModelFactory):
    def create_model(self):
        return SwinUNETR(
            in_channels=3,
            out_channels=self.num_classes,
            img_size=(256, 256),
            norm_name="batch",
            spatial_dims=2,
            use_checkpoint=True,
        )


def get_model_factory(name, num_classes, smp_encoder):
    factories = {
        "UNet": UNetFactory,
        "DeepLabV3": DeepLabV3Factory,
        "DeepLabV3Plus": DeepLabV3PlusFactory,
        "FCN8s": CustomFCN8sFactory,
        "CustomUNet": CustomUNetFactory,
        "CustomResUNet": CustomResUNetFactory,
        "AttResUNet": AttResUNetFactory,
        "AttentionUNet": AttentionUNetFactory,
        "FPN": FPNFactory,
        "UNETR": UNETRFactory,
        "SwinUNETR": SwinUNETRFactory,
        "SmpResUNet": SmpResUNetFactory,
        "SmpResUNetPlusPlus": SmpResUNetPlusPlusFactory,
    }
    if name in factories:
        return factories[name](num_classes, smp_encoder)
    else:
        raise NotImplementedError(f"{name} is not implemented")
