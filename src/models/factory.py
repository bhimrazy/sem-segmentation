import segmentation_models_pytorch as smp
from monai.networks.nets import UNet


class BaseModelFactory:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def create_model(self, name):
        raise NotImplementedError


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
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes,
        )


class DeepLabV3PlusFactory(BaseModelFactory):
    def create_model(self):
        return smp.DeepLabV3Plus(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.num_classes,
        )


def get_model_factory(name, num_classes):
    factories = {
        "UNet": UNetFactory,
        "DeepLabV3": DeepLabV3Factory,
        "DeepLabV3Plus": DeepLabV3PlusFactory,
    }
    if name in factories:
        return factories[name](num_classes)
    else:
        raise NotImplementedError(f"{name} is not implemented")
