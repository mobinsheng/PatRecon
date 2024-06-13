"""
UNet
"""
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


class V4(nn.Module):
    """

    """

    def __init__(self, input_channels, output_channels):
        super(V4, self).__init__()

        short_cut_channels = 128

        self.enc_layer0 = EncoderLayer(input_channels, 128, resnet=True)
        self.enc_layer1 = EncoderLayer(128, 256, resnet=True)
        self.enc_layer2 = EncoderLayer(256, 512, resnet=True)
        self.enc_layer3 = EncoderLayer(512, 1024, resnet=True)
        self.enc_layer4 = EncoderLayer(1024, 2048, resnet=True)
        self.enc_layer5 = EncoderLayer(2048, 4096, resnet=True)

        self.link_layer0 = LinkBlock(128, short_cut_channels, 64, 64)
        self.link_layer1 = LinkBlock(256, short_cut_channels, 32, 32)
        self.link_layer2 = LinkBlock(512, short_cut_channels, 16, 16)

        self.transformer_layer = nn.Sequential(
            Conv2D1x1Block(4096, 4096),
            ConvertFrom2DTo3D(),
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

    def run_resnet(self, data):
        pre_enc_layer_out0, enc0 = self.enc_layer0(data)

        link0 = self.link_layer0(pre_enc_layer_out0)

        pre_enc_layer_out1, enc1 = self.enc_layer1(enc0 + pre_enc_layer_out0)

        link1 = self.link_layer1(pre_enc_layer_out1)

        pre_enc_layer_out2, enc2 = self.enc_layer2(enc1 + pre_enc_layer_out1)

        link2 = self.link_layer2(pre_enc_layer_out2)

        pre_enc_layer_out3, enc3 = self.enc_layer3(enc2 + pre_enc_layer_out2)

        pre_enc_layer_out4, enc4 = self.enc_layer4(enc3 + pre_enc_layer_out3)

        pre_enc_layer_out5, enc5 = self.enc_layer5(enc4)

        features = self.transformer_layer(enc5 + pre_enc_layer_out5)

        dec5 = self.dec_layer5(features)
        dec4 = self.dec_layer4(dec5)
        dec3 = self.dec_layer3(dec4)

        dec2 = self.dec_layer2(torch.cat((dec3, link2), dim=1))
        dec1 = self.dec_layer1(torch.cat((dec2, link1), dim=1))
        dec0 = self.dec_layer0(torch.cat((dec1, link0), dim=1))

        output = self.output_layer(dec0)

        output = torch.squeeze(output, 1)

        return output

    @time_cost
    def forward(self, data):
        return self.run_resnet(data)


if __name__ == "__main__":
    m = V4(3, 128)

    print(m)

    images = torch.randn((1, 3, 128, 128))

    output = m(images)
