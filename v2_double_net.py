"""
UNet
"""
import torch

from util.util import *

from conv_block import *


class EncoderLayer(nn.Module):
    def __init__(self, inc, outc, resnet=False):
        super(EncoderLayer, self).__init__()
        self.resnet = resnet
        self.c1 = Conv2DDownSampleBlock(inc, outc)
        self.c2 = Conv2DBlock(outc, outc)
        pass

    def forward(self, data):
        if self.resnet:
            out1 = self.c1(data)
            out2 = self.c2(out1)
            return out1, out2
        else:
            out1 = self.c1(data)
            out2 = self.c2(out1)
            return out2

    pass


class DecoderLayer(nn.Module):
    """
    解码器部分一般不用ResNet
    """

    def __init__(self, inc, outc):
        super(DecoderLayer, self).__init__()
        self.c1 = ConvTranspose3dUpSampleBlock(inc, outc)
        self.c2 = Transpose3D1x1Block(outc, outc)  # ConvTranspose3dBlock(outc, outc),
        pass

    def forward(self, data):
        out1 = self.c1(data)
        out2 = self.c2(out1)
        return out2

    pass


class X2CTGenerator(nn.Module):
    """

    """

    def __init__(self, input_channels, output_channels):
        super(X2CTGenerator, self).__init__()

        short_cut_channels = 128

        self.use_unet = True

        if not self.use_unet:
            short_cut_channels = 0

        self.enc_layer0 = EncoderLayer(input_channels, 128, resnet=True)
        self.enc_layer1 = EncoderLayer(128, 256, resnet=True)
        self.enc_layer2 = EncoderLayer(256, 512, resnet=True)
        self.enc_layer3 = EncoderLayer(512, 1024, resnet=True)
        self.enc_layer4 = EncoderLayer(1024, 2048, resnet=True)
        self.enc_layer5 = EncoderLayer(2048, 4096, resnet=True)

        if self.use_unet:
            self.link_layer0 = LinkBlock(128, short_cut_channels, 64, 64)
            self.link_layer1 = LinkBlock(256, short_cut_channels, 32, 32)
            self.link_layer2 = LinkBlock(512, short_cut_channels, 16, 16)

        self.transformer_layer = nn.Sequential(
            Conv2D1x1Block(4096, 4096),
            To3D(),
            Transpose3D1x1Block(2048, 2048),

        )

        self.dec_layer5 = DecoderLayer(2048, 1024)
        self.dec_layer4 = DecoderLayer(1024, 512)
        self.dec_layer3 = DecoderLayer(512, 256)
        self.dec_layer2 = DecoderLayer(256 + short_cut_channels, 128)
        self.dec_layer1 = DecoderLayer(128 + short_cut_channels, 64)
        self.dec_layer0 = DecoderLayer(64 + short_cut_channels, 16)

        self.output_layer = OutputBlock(16, 1)

        initialize_weights(self)

        pass

    def cat(self, out, link):
        if self.use_unet:
            return torch.cat((out, link), dim=1)
        else:
            return out

    def run_resnet(self, data):
        pre_enc_layer_out0, enc0 = self.enc_layer0(data)



        pre_enc_layer_out1, enc1 = self.enc_layer1(enc0 + pre_enc_layer_out0)


        pre_enc_layer_out2, enc2 = self.enc_layer2(enc1 + pre_enc_layer_out1)


        pre_enc_layer_out3, enc3 = self.enc_layer3(enc2 + pre_enc_layer_out2)

        pre_enc_layer_out4, enc4 = self.enc_layer4(enc3 + pre_enc_layer_out3)

        pre_enc_layer_out5, enc5 = self.enc_layer5(enc4 + pre_enc_layer_out4)

        link0 = None
        link1 = None
        link2 = None

        if self.use_unet:
            link0 = self.link_layer0(pre_enc_layer_out0)
            link1 = self.link_layer1(pre_enc_layer_out1)
            link2 = self.link_layer2(pre_enc_layer_out2)

        features = self.transformer_layer(enc5 + pre_enc_layer_out5)

        dec5 = self.dec_layer5(features)
        dec4 = self.dec_layer4(dec5)
        dec3 = self.dec_layer3(dec4)

        dec2 = self.dec_layer2(self.cat(dec3, link2))
        dec1 = self.dec_layer1(self.cat(dec2, link1))
        dec0 = self.dec_layer0(self.cat(dec1, link0))

        output = self.output_layer(dec0)

        output = torch.squeeze(output, 1)

        return output

    @time_cost
    def forward(self, data):
        return self.run_resnet(data)


class CT2XGenerator(nn.Module):
    """

    """

    def __init__(self, input_channels, output_channels):
        super(CT2XGenerator, self).__init__()

        self.model = nn.Sequential(
            Conv3dDownSampleBlock(1, 32),
            Conv3dDownSampleBlock(32, 64),
            Conv3dDownSampleBlock(64, 128),
            Conv3dDownSampleBlock(128, 256),
            Conv3dDownSampleBlock(256, 512),
            Conv3dDownSampleBlock(512, 1024),  # [1024, 2, 2, 2]

            To2D(),

            Conv2DBlock(2048, 2048),
            ConvTranspose2dUpSampleBlock(2048, 1024),
            Conv2DBlock(1024, 1024),
            ConvTranspose2dUpSampleBlock(1024, 512),
            Conv2DBlock(512, 512),
            ConvTranspose2dUpSampleBlock(512, 256),
            Conv2DBlock(256, 256),
            ConvTranspose2dUpSampleBlock(256, 128),
            Conv2DBlock(128, 128),
            ConvTranspose2dUpSampleBlock(128, 64),
            Conv2DBlock(64, 64),
            ConvTranspose2dUpSampleBlock(64, 32),
            nn.Conv2d(in_channels=32,
                      out_channels=3,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
        )

        initialize_weights(self)

        pass

    def cat(self, out, link):
        if self.use_unet:
            return torch.cat((out, link), dim=1)
        else:
            return out

    @time_cost
    def forward(self, data):
        output = self.model(data)
        return output


class CTLoss(nn.Module):
    def __init__(self, inc, outc):
        super(CTLoss, self).__init__()
        self.model = CT2XGenerator(inc, outc)
        self.criterion = nn.MSELoss(reduction='mean').to(self.device)
        pass

    def forward(self, ct, x_rays):
        out = self.model(ct)
        return self.criterion(ct, x_rays)


if __name__ == "__main__":
    """m = X2CTGenerator(3, 128)

    print(m)

    images = torch.randn((1, 3, 128, 128))

    output = m(images)"""

    m = CT2XGenerator(1, 3)
    print(m)
    images = torch.randn((1, 1, 128, 128, 128))
    output = m(images)
    print(output.shape)

    """m = DenseBlock(16, 28, 8, 4)
    print(m)
    images = torch.randn((1, 16, 128, 128))

    output = m(images)
    print(output.shape)"""
