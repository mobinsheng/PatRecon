import torch.nn as nn
import numpy as np
import torch
import math

"""
模仿ReconNet的网络结构，修改点有下面几个：
1、生成网络的层数不一样
2、从2d变成3d的时候，原来的网络的起始分辨率是2x4x4，这里变成4x4x4
3、去掉最后输出模块的里面的二维卷积
"""
from util.util import *


class Conv2DBlock(nn.Module):
    """
    2维卷积，分辨率不变，只用于提取特征
    """

    def __init__(self, input_channels, output_channels):
        super(Conv2DBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)

        self.norm = nn.BatchNorm2d(output_channels)

        pass

    def forward(self, data):
        output = self.conv(data)
        output = self.norm(output)
        return output

    pass


class Conv2DDownSampleBlock(nn.Module):
    """
    通过2维卷积实现分辨率减半
    """

    def __init__(self, input_channels, output_channels):
        super(Conv2DDownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = nn.BatchNorm2d(output_channels)
        pass

    def forward(self, data):
        output = self.conv(data)
        output = self.norm(output)
        return output

    pass


class ConvTranspose3dBlock(nn.Module):
    """
    3维卷积，分辨率不变，只用于提取特征
    """

    def __init__(self, input_channels, output_channels):
        super(ConvTranspose3dBlock, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels=input_channels,
                                       out_channels=output_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)
        self.norm = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, data):
        output = self.conv(data)
        output = self.norm(output)
        output = self.relu(output)
        return output

    pass


class ConvTranspose3dUpSampleBlock(nn.Module):
    """
    通过3维卷积实现分辨率加倍
    """

    def __init__(self, input_channels, output_channels):
        super(ConvTranspose3dUpSampleBlock, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels=input_channels,
                                       out_channels=output_channels,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1,
                                       bias=False)
        self.norm = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, data):
        output = self.conv(data)
        output = self.norm(output)
        output = self.relu(output)
        return output

    pass


class Transpose2DBlock(nn.Module):
    """
    通过3维卷积实现分辨率加倍
    """

    def __init__(self, input_channels, output_channels):
        super(Transpose2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False)
        self.norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, data):
        output = self.conv(data)
        output = self.norm(output)
        output = self.relu(output)
        return output

    pass


class Transpose3DBlock(nn.Module):
    """
    通过3维卷积实现分辨率加倍
    """

    def __init__(self, input_channels, output_channels):
        super(Transpose3DBlock, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels=input_channels,
                                       out_channels=output_channels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)
        self.norm = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        pass

    def forward(self, data):
        output = self.conv(data)
        output = self.norm(output)
        output = self.relu(output)
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
        assert width == 4

        depth = 4
        new_channels = channels // 4

        # 升维，从[N, C, H, W] 增加了一个Depth维度，变成 [N, C/4, 4, H, W]
        # Depth的大小是4
        output = data.view(batch_size, new_channels, depth, height, width)
        return output

    pass


class OutputBlock(nn.Module):
    """
    输出模块
    经过一个1x1的卷积层，类似于于一个全连接层
    """

    def __init__(self, input_channels, outpu_channels):
        super(OutputBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels=input_channels,
                      out_channels=4,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Conv3d(in_channels=4,
                      out_channels=2,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Conv3d(in_channels=2,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False)
        )
        """self.conv3d = nn.Conv3d(in_channels=input_channels,
                                out_channels=1,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)"""
        pass

    def forward(self, data):
        #output = self.conv3d(data)
        output = self.model(data)
        print_shape(output)
        return output


def print_shape(data):
    #print("shape:", data.shape)
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


class X2CT(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(X2CT, self).__init__()

        self.encoder_layer1 = Conv2DDownSampleBlock(input_channels, 64)
        self.encoder_relu1 = nn.ReLU(inplace=True)
        self.encoder_layer2 = Conv2DBlock(64, 64)
        self.encoder_relu2 = nn.ReLU(inplace=True)

        self.encoder_layer3 = Conv2DDownSampleBlock(64, 128)
        self.encoder_relu3 = nn.ReLU(inplace=True)
        self.encoder_layer4 = Conv2DBlock(128, 128)
        self.encoder_relu4 = nn.ReLU(inplace=True)

        self.encoder_layer5 = Conv2DDownSampleBlock(128, 256)
        self.encoder_relu5 = nn.ReLU(inplace=True)
        self.encoder_layer6 = Conv2DBlock(256, 256)
        self.encoder_relu6 = nn.ReLU(inplace=True)

        self.encoder_layer7 = Conv2DDownSampleBlock(256, 512)
        self.encoder_relu7 = nn.ReLU(inplace=True)
        self.encoder_layer8 = Conv2DBlock(512, 512)
        self.encoder_relu8 = nn.ReLU(inplace=True)

        self.encoder_layer9 = Conv2DDownSampleBlock(512, 1024)
        self.encoder_relu9 = nn.ReLU(inplace=True)
        self.encoder_layer10 = Conv2DBlock(1024, 1024)
        self.encoder_relu10 = nn.ReLU(inplace=True)

        self.trans_layer1 = Transpose2DBlock(1024, 1024)
        self.convert_3d = ConvertFrom2DTo3D()
        self.trans_layer2 = Transpose3DBlock(256, 256)

        self.decoder_layer10 = ConvTranspose3dUpSampleBlock(256, 128)
        self.decoder_layer9 = ConvTranspose3dBlock(128, 128)

        self.decoder_layer8 = ConvTranspose3dUpSampleBlock(128, 64)
        self.decoder_layer7 = ConvTranspose3dBlock(64, 64)

        self.decoder_layer6 = ConvTranspose3dUpSampleBlock(64, 32)
        self.decoder_layer5 = ConvTranspose3dBlock(32, 32)

        self.decoder_layer4 = ConvTranspose3dUpSampleBlock(32, 16)
        self.decoder_layer3 = ConvTranspose3dBlock(16, 16)

        self.decoder_layer2 = ConvTranspose3dUpSampleBlock(16, 8)
        self.decoder_layer1 = ConvTranspose3dBlock(8, 8)

        self.decoder_layer = OutputBlock(8, 1)

        _initialize_weights(self)

        pass

    def run_res_net(self, data):
        print_shape(data)

        c1 = self.encoder_layer1(data)
        r1 = self.encoder_relu1(c1)
        c2 = self.encoder_layer2(r1)
        r2 = self.encoder_relu2(c2 + r1)

        c3 = self.encoder_layer3(r2)
        r3 = self.encoder_relu3(c3)
        c4 = self.encoder_layer4(r3)
        r4 = self.encoder_relu4(c4 + r3)

        c5 = self.encoder_layer5(r4)
        r5 = self.encoder_relu5(c5)
        c6 = self.encoder_layer6(r5)
        r6 = self.encoder_relu6(c6 + r5)

        c7 = self.encoder_layer7(r6)
        r7 = self.encoder_relu7(c7)
        c8 = self.encoder_layer8(r7)
        r8 = self.encoder_relu8(c8 + r7)

        c9 = self.encoder_layer9(r7)
        r9 = self.encoder_relu9(c9)
        c10 = self.encoder_layer10(r9)
        r10 = self.encoder_relu10(c10 + r9)

        features = self.trans_layer1(r10)
        features = self.convert_3d(features)
        features = self.trans_layer2(features)

        dc10 = self.decoder_layer10(features)
        dc9 = self.decoder_layer9(dc10)

        dc8 = self.decoder_layer8(dc9)
        dc7 = self.decoder_layer7(dc8)

        dc6 = self.decoder_layer6(dc7)
        dc5 = self.decoder_layer5(dc6)

        dc4 = self.decoder_layer4(dc5)
        dc3 = self.decoder_layer3(dc4)

        dc2 = self.decoder_layer2(dc3)
        dc1 = self.decoder_layer1(dc2)

        output = self.decoder_layer(dc1)

        output = torch.squeeze(output, 1)

        return output


    def run_simple(self, data):
        print_shape(data)

        c1 = self.encoder_layer1(data)
        r1 = self.encoder_relu1(c1)

        c3 = self.encoder_layer3(r1)
        r3 = self.encoder_relu3(c3)

        c5 = self.encoder_layer5(r3)
        r5 = self.encoder_relu5(c5)

        c7 = self.encoder_layer7(r5)
        r7 = self.encoder_relu7(c7)

        c9 = self.encoder_layer9(r7)
        r9 = self.encoder_relu9(c9)

        features = self.trans_layer1(r9)
        features = self.convert_3d(features)
        features = self.trans_layer2(features)

        dc10 = self.decoder_layer10(features)
        dc9 = self.decoder_layer9(dc10)

        dc8 = self.decoder_layer8(dc9)
        dc7 = self.decoder_layer7(dc8)

        dc6 = self.decoder_layer6(dc7)
        dc5 = self.decoder_layer5(dc6)

        dc4 = self.decoder_layer4(dc5)
        dc3 = self.decoder_layer3(dc4)

        dc2 = self.decoder_layer2(dc3)
        dc1 = self.decoder_layer1(dc2)

        output = self.decoder_layer(dc1)

        output = torch.squeeze(output, 1)

        return output
    @time_cost
    def forward(self, data):
        return self.run_res_net(data)


if __name__ == "__main__":
    m = X2CT(3, 128)

    print(m)

    images = torch.randn((1, 3, 128, 128))

    output = m(images)
