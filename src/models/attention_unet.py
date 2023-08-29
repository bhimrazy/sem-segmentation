import torch
import torch.nn as nn
from torchvision import transforms


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


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
            self.encoder_blocks.append(ConvBlock(channels[i], channels[i + 1]))

    def forward(self, x):
        # print("Contracting...")
        features = []
        for enc in self.encoder_blocks:
            x = enc(x)
            features.append(x)
            # print(x.shape)
            x = self.max_pool(x)

        return features


class AttentionBlock(nn.Module):
    def __init__(self, g_channels, xl_channels):
        """
        Args:
        g_channels : channels of lower gating/decoder block
        xl_channles: channels of encoder input feature
        """
        super(AttentionBlock, self).__init__()

        self.conv_g = nn.Conv2d(
            in_channels=g_channels, out_channels=xl_channels, kernel_size=1
        )
        self.conv_xl = nn.Conv2d(
            in_channels=xl_channels, out_channels=xl_channels, kernel_size=1, stride=2
        )
        self.relu = nn.ReLU()

        self.conv_psi = nn.Conv2d(
            in_channels=xl_channels, out_channels=1, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.att_conv = nn.Conv2d(
            in_channels=xl_channels, out_channels=xl_channels, kernel_size=1
        )
        self.bn = nn.BatchNorm2d(xl_channels)

    def forward(self, g, xl):
        # g: Input feature map from (decoder path)
        # xl(encoder_feature): Corresponding feature map from the encoder path

        conv_g = self.conv_g(g)
        conv_xl = self.conv_xl(xl)

        xg = conv_g + conv_xl
        xg = self.relu(xg)
        xg = self.conv_psi(xg)
        xg = self.sigmoid(xg)
        xg = self.upsample(xg)

        att = xg * xl
        att = self.att_conv(att)
        att = self.bn(att)
        return att


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
        self.attention_blocks = nn.ModuleList()

        for i in range(len(channels) - 1):
            self.up_convs.append(nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2))
            self.decoder_blocks.append(ConvBlock(channels[i], channels[i + 1]))

        for i in range(len(channels) - 1):
            self.attention_blocks.append(AttentionBlock(channels[i], channels[i + 1]))

    def crop(self, feature, x):
        _, _, H, W = x.shape
        feature = transforms.CenterCrop([H, W])(feature)
        return feature

    def forward(self, x, encoder_features):
        # print("Expanding...")
        for i, (up_conv, dec, attention_block) in enumerate(
            zip(self.up_convs, self.decoder_blocks, self.attention_blocks)
        ):
            att_feature = attention_block(x, encoder_features[i])
            x = up_conv(x)
            # encoder_feature = self.crop(att_feature,x)
            x = torch.cat([att_feature, x], dim=1)
            x = dec(x)

            # print(x.shape)

        return x


class AttentionUNet(nn.Module):
    # AttResUNet implementation in pytorch
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024), num_classes=1):
        super(AttentionUNet, self).__init__()
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
