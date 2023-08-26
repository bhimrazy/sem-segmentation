import torch.nn as nn


class ConvBlock(nn.Module):
    """This block consists of:
    - Two consecutive CNN layers (each => conv 3x3, ReLU)
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
            nn.Conv2d(
                in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FCN8s(nn.Module):
    def __init__(self, num_classes=1):
        super(FCN8s, self).__init__()

        self.conv_1 = nn.Sequential(
            ConvBlock(3, 64), nn.MaxPool2d(kernel_size=2)  # 128
        )
        self.conv_2 = nn.Sequential(
            ConvBlock(64, 128), nn.MaxPool2d(kernel_size=2)  # 64
        )
        self.conv_3 = nn.Sequential(
            ConvBlock(128, 256), nn.MaxPool2d(kernel_size=2)  # 32
        )
        self.conv_4 = nn.Sequential(
            ConvBlock(256, 512), nn.MaxPool2d(kernel_size=2)  # 16
        )
        self.conv_5 = nn.Sequential(
            ConvBlock(512, 1024),
            nn.MaxPool2d(kernel_size=2),  # 8
        )

        self.up_conv_1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 768, 2, 2),  # 16
            ConvBlock(768, 512),
            ConvBlock(512, 512),
        )
        self.pool_4_pred = nn.Conv2d(512, 512, kernel_size=1)
        self.up_conv_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 384, 2, 2),  # 32
            ConvBlock(384, 256),
            ConvBlock(256, 256),
        )
        self.pool_3_pred = nn.Conv2d(256, 256, kernel_size=1)

        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=16, stride=8, padding=4),  # 256
            ConvBlock(64, 16),
            ConvBlock(16, 16),
        )

        self.out = nn.Sequential(nn.Conv2d(16, 1, 1), nn.Sigmoid())  # 256

    def forward(self, x):
        # x : 1 x 256 x 256

        # encoder
        conv_1x = self.conv_1(x)  # 64 x 128 x 128
        conv_2x = self.conv_2(conv_1x)  # 128 x 64 x 64
        conv_3x = self.conv_3(conv_2x)  # 256 x 32 x 32
        conv_4x = self.conv_4(conv_3x)  # 512 x 16 x 16
        conv_5x = self.conv_5(conv_4x)  # 1024 x 8 x 8

        # decoder
        up_conv1 = self.up_conv_1(conv_5x)  # 512 x 16 x 16
        # pool_4   = self.pool_4_pred(conv_4x)  # 512 x 16 x 16
        up_conv_2x = up_conv1 + conv_4x  # pool_4        # 512 x 16 x 16

        up_conv2 = self.up_conv_2(up_conv_2x)  # 256 x 32 x 32
        # pool_3   = self.pool_3_pred(conv_3x)  # 256 x 32 x 32
        up_conv_4x = up_conv2 + conv_3x  # pool_3        # 256 x 32 x 32

        up_conv_8x = self.conv_transpose(up_conv_4x)  # 1 x 256 x 256

        out = self.out(up_conv_8x)
        return out
