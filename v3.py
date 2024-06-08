import torch.nn as nn
import numpy as np
import torch
import math

"""
UNet
"""
#from util.util import *


class Conv2DBlock(nn.Module):
    """
    2维卷积，分辨率不变，只用于提取特征
    """

    def __init__(self, input_channels, output_channels):
        super(Conv2DBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),

            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

        )

        pass

    def forward(self, data):
        output = self.model(data)
        return output

    pass


class Conv2DDownSampleBlock(nn.Module):
    """
    通过2维卷积实现分辨率减半
    """

    def __init__(self, input_channels, output_channels):
        super(Conv2DDownSampleBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

        )

        pass

    def forward(self, data):
        output = self.model(data)
        return output

    pass


class ConvTranspose3dBlock(nn.Module):
    """
    3维卷积，分辨率不变，只用于提取特征
    """

    def __init__(self, input_channels, output_channels):
        super(ConvTranspose3dBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_channels=input_channels,
                               out_channels=output_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),

        )
        pass

    def forward(self, data):
        output = self.model(data)
        return output

    pass


class ConvTranspose3dUpSampleBlock(nn.Module):
    """
    通过3维卷积实现分辨率加倍
    """

    def __init__(self, input_channels, output_channels):
        super(ConvTranspose3dUpSampleBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_channels=input_channels,
                               out_channels=output_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),

        )
        pass

    def forward(self, data):
        output = self.model(data)
        return output

    pass


class Conv2D1x1Block(nn.Module):
    """
    1x1的卷积，用于改变channel数量
    """

    def __init__(self, input_channels, output_channels):
        super(Conv2D1x1Block, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=output_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        pass

    def forward(self, data):
        output = self.model(data)
        return output

    pass


class Transpose3D1x1Block(nn.Module):
    """
    1x1的卷积，用于改变channel的数量
    """

    def __init__(self, input_channels, output_channels):
        super(Transpose3D1x1Block, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(in_channels=input_channels,
                               out_channels=output_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),

        )
        pass

    def forward(self, data):
        output = self.model(data)
        return output

    pass


class ConvertFrom2DTo3D(nn.Module):
    """
    数据从2D变为3D
    具体的：
    1、数据从[N, C, H, W]变成[N, C, D, H, W]
    2、D是新增的维度：深度
    """

    def __init__(self):
        super(ConvertFrom2DTo3D, self).__init__()
        pass

    def forward(self, data):
        batch_size, channels, height, width = data.shape

        assert height == width
        # assert width == 2

        depth = width
        new_channels = channels // depth

        # 升维，从[N, C, H, W] 增加了一个Depth维度，变成 [N, C/4, 4, H, W]
        # Depth的大小是4
        output = data.view(batch_size, new_channels, depth, height, width)
        return output

    pass


class LinkBlock(nn.Module):
    """
    UNet的link模块
    """

    def __init__(self, input_channels, output_channels, width, height):
        super(LinkBlock, self).__init__()

        assert width == height
        assert width % 2 == 0

        self.depth = width

        middle_channels = self.depth

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=middle_channels * input_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(middle_channels * input_channels),
            nn.ReLU(inplace=True),

            ConvertFrom2DTo3D(),

            nn.ConvTranspose3d(in_channels=input_channels,
                               out_channels=output_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),

        )
        pass

    def forward(self, data):
        """output = data.unsqueeze(2)
        output = output.repeat(1,1, self.depth, 1, 1)
        return output"""

        output = self.model(data)
        return output

    pass


class OutputBlock(nn.Module):
    """
    输出模块
    经过一个1x1的卷积层，类似于于一个全连接层
    """

    def __init__(self, input_channels, output_channels):
        super(OutputBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels=input_channels,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False)
        )
        pass

    def forward(self, data):
        # output = self.conv3d(data)
        output = self.model(data)
        print_shape(output)
        return output


def print_shape(data):
    # print("shape:", data.shape)
    pass


def _initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class V3(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(V3, self).__init__()

        self.enable_unet = True
        self.multiple = 1

        if self.enable_unet:
            self.multiple = 2

        self.base_conv2d0 = Conv2DBlock(input_channels,1024)
        self.down_sample_conv2d0 = Conv2DDownSampleBlock(1024, 1024)

        self.link_layer0 = LinkBlock(1024, 32, 64, 64)

        self.base_conv2d1 = Conv2DBlock(1024, 1024)
        self.down_sample_conv2d1 = Conv2DDownSampleBlock(1024, 1024)

        self.link_layer1 = LinkBlock(1024, 64, 32, 32)

        self.base_conv2d2 = Conv2DBlock(1024, 1024)
        self.down_sample_conv2d2 = Conv2DDownSampleBlock(1024, 1024)

        self.link_layer2 = LinkBlock(1024, 128, 16, 16)

        self.base_conv2d3 = Conv2DBlock(1024, 1024)
        self.down_sample_conv2d3 = Conv2DDownSampleBlock(1024, 1024)

        self.link_layer3 = LinkBlock(1024, 256, 8, 8)

        self.base_conv2d4 = Conv2DBlock(1024, 1024)
        self.down_sample_conv2d4 = Conv2DDownSampleBlock(1024, 1024)

        self.link_layer4 = LinkBlock(1024, 512, 4, 4)

        self.base_conv2d5 = Conv2DBlock(1024, 2048)
        self.down_sample_conv2d5 = Conv2DDownSampleBlock(2048, 2048)

        self.link_layer5 = LinkBlock(2048, 1024, 2, 2)

        self.transformer_layer = nn.Sequential(
            Conv2D1x1Block(2048, 2048),
            ConvertFrom2DTo3D(),
            Transpose3D1x1Block(1024, 1024),

        )

        self.up_sample_conv3d5 = ConvTranspose3dUpSampleBlock(1024 * self.multiple, 512)
        self.up_sample_conv3d4 = ConvTranspose3dUpSampleBlock(512 * self.multiple, 256)
        self.up_sample_conv3d3 = ConvTranspose3dUpSampleBlock(256 * self.multiple, 128)
        self.up_sample_conv3d2 = ConvTranspose3dUpSampleBlock(128 * self.multiple, 64)
        self.up_sample_conv3d1 = ConvTranspose3dUpSampleBlock(64 * self.multiple, 32)
        self.up_sample_conv3d0 = ConvTranspose3dUpSampleBlock(32 * self.multiple, 16)

        self.output_layer = OutputBlock(16, 1)

        _initialize_weights(self)

        pass

    def run(self, data):
        enc0 = self.down_sample_conv2d0(self.base_conv2d0(data))

        enc1 = self.down_sample_conv2d1(self.base_conv2d1(enc0))

        enc2 = self.down_sample_conv2d2(self.base_conv2d2(enc1))

        enc3 = self.down_sample_conv2d3(self.base_conv2d3(enc2))

        enc4 = self.down_sample_conv2d4(self.base_conv2d4(enc3))

        enc5 = self.down_sample_conv2d5(self.base_conv2d5(enc4))

        features = self.transformer_layer(enc5)

        dec5 = self.up_sample_conv3d5(features)
        dec4 = self.up_sample_conv3d4(dec5)
        dec3 = self.up_sample_conv3d3(dec4)
        dec2 = self.up_sample_conv3d2(dec3)
        dec1 = self.up_sample_conv3d1(dec2)
        dec0 = self.up_sample_conv3d0(dec1)

        output = self.output_layer(dec0)

        output = torch.squeeze(output, 1)

        return output

    def run_unet(self, data):
        enc0 = self.down_sample_conv2d0(self.base_conv2d0(data))
        link0 = self.link_layer0(enc0)

        enc1 = self.down_sample_conv2d1(self.base_conv2d1(enc0))
        link1 = self.link_layer1(enc1)

        enc2 = self.down_sample_conv2d2(self.base_conv2d2(enc1))
        link2 = self.link_layer2(enc2)

        enc3 = self.down_sample_conv2d3(self.base_conv2d3(enc2))
        link3 = self.link_layer3(enc3)

        enc4 = self.down_sample_conv2d4(self.base_conv2d4(enc3))
        link4 = self.link_layer4(enc4)

        enc5 = self.down_sample_conv2d5(self.base_conv2d5(enc4))
        link5 = self.link_layer5(enc5)

        features = self.transformer_layer(enc5)

        dec5 = self.up_sample_conv3d5(torch.cat((features, link5), dim=1))
        dec4 = self.up_sample_conv3d4(torch.cat((dec5, link4), dim=1))
        dec3 = self.up_sample_conv3d3(torch.cat((dec4, link3), dim=1))
        dec2 = self.up_sample_conv3d2(torch.cat((dec3, link2), dim=1))
        dec1 = self.up_sample_conv3d1(torch.cat((dec2, link1), dim=1))
        dec0 = self.up_sample_conv3d0(torch.cat((dec1, link0), dim=1))

        output = self.output_layer(dec0)

        output = torch.squeeze(output, 1)

        return output

    #@time_cost
    def forward(self, data):
        if self.enable_unet:
            return self.run_unet(data)
        else:
            return self.run(data)


if __name__ == "__main__":
    """m = V3(3, 128)

    print(m)

    images = torch.randn((1, 3, 128, 128))

    output = m(images)"""
    data = torch.randn(1,1,4,4)
    data = data.unsqueeze(2)
    print(data.shape)
    data = data.repeat(1,1,4,1,1)
    print(data.shape)
