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
                              kernel_size=4,
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
        self.conv3d = nn.Conv3d(in_channels=input_channels,
                                out_channels=1,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False)
        pass

    def forward(self, data):
        output = self.conv3d(data)
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

        self.encoder_layer1 = Conv2DDownSampleBlock(input_channels, 256)
        self.encoder_relu1 = nn.ReLU(inplace=True)

        self.encoder_layer3 = Conv2DDownSampleBlock(256, 512)
        self.encoder_relu3 = nn.ReLU(inplace=True)

        self.encoder_layer5 = Conv2DDownSampleBlock(512, 1024)
        self.encoder_relu5 = nn.ReLU(inplace=True)

        self.encoder_layer7 = Conv2DDownSampleBlock(1024, 2048)
        self.encoder_relu7 = nn.ReLU(inplace=True)

        self.encoder_layer9 = Conv2DDownSampleBlock(2048, 4096)
        self.encoder_relu9 = nn.ReLU(inplace=True)

        self.trans_layer1 = Transpose2DBlock(4096, 4096)
        self.convert_3d = ConvertFrom2DTo3D()
        self.trans_layer2 = Transpose3DBlock(1024, 1024)

        self.decoder_layer10 = ConvTranspose3dUpSampleBlock(1024, 512)

        self.decoder_layer8 = ConvTranspose3dUpSampleBlock(512, 256)

        self.decoder_layer6 = ConvTranspose3dUpSampleBlock(256, 128)

        self.decoder_layer4 = ConvTranspose3dUpSampleBlock(128, 64)

        self.decoder_layer2 = ConvTranspose3dUpSampleBlock(64, 32)

        self.decoder_layer = OutputBlock(32, 1)

        _initialize_weights(self)

        pass

    def forward(self, data):
        print_shape(data)

        c1 = self.encoder_layer1(data)
        r1 = self.encoder_relu1(c1)
        print_shape(r1)

        c3 = self.encoder_layer3(r1)
        r3 = self.encoder_relu3(c3)
        print_shape(r3)

        c5 = self.encoder_layer5(r3)
        r5 = self.encoder_relu5(c5)
        print_shape(r5)

        c7 = self.encoder_layer7(r5)
        r7 = self.encoder_relu7(c7)
        print_shape(r7)

        c9 = self.encoder_layer9(r7)
        r9 = self.encoder_relu9(c9)
        print_shape(r9)

        features = self.trans_layer1(r9)
        print_shape(features)
        features = self.convert_3d(features)
        print_shape(features)
        features = self.trans_layer2(features)
        print_shape(features)

        dc10 = self.decoder_layer10(features)
        print_shape(dc10)

        dc8 = self.decoder_layer8(dc10)
        print_shape(dc8)

        dc6 = self.decoder_layer6(dc8)
        print_shape(dc6)

        dc4 = self.decoder_layer4(dc6)
        print_shape(dc4)

        dc2 = self.decoder_layer2(dc4)
        print_shape(dc2)

        output = self.decoder_layer(dc2)

        return output


if __name__ == "__main__":
    m = X2CT(3, 128)

    print(m)

    images = torch.randn((1, 3, 128, 128))

    output = m(images)