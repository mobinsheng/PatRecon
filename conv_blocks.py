import torch.nn as nn
import numpy as np
import torch
import math


def print_shape(data):
    # print("shape:", data.shape)
    pass


def initialize_weights(net):
    """
    权重初始化
    :param net: 
    :return: 
    """
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
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1,
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
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(middle_channels * input_channels),
            nn.ReLU(inplace=True),

            ConvertFrom2DTo3D(),

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
            nn.ConvTranspose3d(in_channels=input_channels,
                               out_channels=1,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        )

        """
        count = round(math.sqrt(input_channels))

        nc = input_channels
        
        for _ in range(count):

            if nc // 2 == 1:
                conv = nn.ConvTranspose3d(in_channels=nc,
                                          out_channels=nc // 2,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=False)
                self.model.append(conv)
            else:
                conv = ConvTranspose3dBlock(nc, nc // 2)
                self.model.append(conv)
            nc //= 2
        """
        pass

    def forward(self, data):
        # output = self.conv3d(data)
        output = self.model(data)
        print_shape(output)
        return output
