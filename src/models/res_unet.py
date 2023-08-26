import torch
import torch.nn as nn
from torchvision import transforms


class ResConvBlock(nn.Module):
    """This block consists of:
    - Two consecutive CNN layers (each => conv 3x3, ReLU)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_ch),
        )

        self.skip_conn = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        block_x = self.block(x)

        # shortcut conn
        skip_x = self.skip_conn(x)

        # adding residual
        res_out = skip_x + block_x
        return res_out


class ContractingEncoder(nn.Module):
    """This block consists of:
    - Two consecutive CNN layers (each => conv 3x3, ReLU)
    - Followed by maxpooling (max pool 2x2)
    """

    def __init__(self, channels):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

        for i in range(len(channels) - 1):
            self.encoder_blocks.append(ResConvBlock(channels[i], channels[i + 1]))

    def forward(self, x):
        # print("Contracting...")
        features = []
        for enc in self.encoder_blocks:
            x = enc(x)
            features.append(x)
            # print(x.shape)
            x = self.max_pool(x)

        return features


class ExpandingDecoder(nn.Module):
    """This block consists of:
    - Up sampling conv layer (up-conv 2x2 , makes channel half)
    - Concatenation of corresponding cropped feature from contracting path
    - Two consecutive CNN layers (=> conv 3x3, ReLU)
    """

    def __init__(self, channels):
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.up_convs.append(nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2))
            self.decoder_blocks.append(ResConvBlock(channels[i], channels[i + 1]))

    def crop(self, feature, x):
        _, _, H, W = x.shape
        feature = transforms.CenterCrop([H, W])(feature)
        return feature

    def forward(self, x, encoder_features):
        # print("Expanding...")
        for i, (up_conv, dec) in enumerate(zip(self.up_convs, self.decoder_blocks)):
            x = up_conv(x)
            encoder_feature = self.crop(encoder_features[i], x)
            x = torch.cat([encoder_feature, x], dim=1)
            x = dec(x)

            # print(x.shape)

        return x


class ResUNet(nn.Module):
    # UNet implementation in pytorch
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024), num_classes=1):
        super(ResUNet, self).__init__()
        self.enc_channels = channels
        self.dec_channels = channels[::-1][:-1]

        self.encoder = ContractingEncoder(channels=self.enc_channels)
        self.decoder = ExpandingDecoder(channels=self.dec_channels)

        self.head = nn.Conv2d(self.dec_channels[-1], num_classes, 1)

    def forward(self, x):
        encoder_features = self.encoder(x)
        encoder_features = encoder_features[::-1]
        out = self.decoder(encoder_features[0], encoder_features[1:])
        out = self.head(out)
        return out
