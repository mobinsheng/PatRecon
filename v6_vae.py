from __future__ import print_function
import argparse
import random
import shutil

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from data_reader import *
from conv_block import *
from v4_resnet_unet import EncoderLayer, DecoderLayer

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--manualSeed', type=int, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--enable_norm', type=bool, default=False, help='enable_norm')
parser.add_argument('--enable_adaptive_lr', type=bool, default=True, help='enable_adaptive_lr')
parser.add_argument('--train_file', type=str, default="./dataset/test.csv", help='weight_decay')
parser.add_argument('--test_file', type=str, default="./dataset/test.csv", help='weight_decay')
parser.add_argument('--data_root', type=str,
                    default="/Users/mobinsheng/work/other/pytorch-projects/PatRecon/dataset/data", help='data_root')
parser.add_argument('--output_path', type=str, default="./dataset", help='output_path')
parser.add_argument('--resume', type=str, default="final", help='resume')
parser.add_argument('--num-views', type=int, default=3, help='number of views/projections in inputs')
parser.add_argument('--input-size', type=int, default=128, help='dimension of input view size')
parser.add_argument('--output-size', type=int, default=128, help='dimension of ouput 3D model size')
parser.add_argument('--output-channel', type=int, default=128, help='dimension of ouput 3D model size')
args = parser.parse_args()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

train_dataset = XCTDataset(args.enable_norm, args.data_root, args.train_file, args.num_views, args.input_size)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    pin_memory=True)

test_dataset = XCTDataset(args.enable_norm, args.data_root, args.test_file, args.num_views, args.input_size)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True)


class VAE(nn.Module):
    def __init__(self, input_channels):
        super(VAE, self).__init__()

        self.enc_layer0 = EncoderLayer(input_channels, 128, resnet=True)
        self.enc_layer1 = EncoderLayer(128, 256, resnet=True)
        self.enc_layer2 = EncoderLayer(256, 512, resnet=True)
        self.enc_layer3 = EncoderLayer(512, 1024, resnet=True)
        self.enc_layer4 = EncoderLayer(1024, 2048, resnet=True)
        self.enc_layer5 = EncoderLayer(2048, 4096, resnet=True)

        self.transformer_layer = nn.Sequential(
            Conv2D1x1Block(4096, 4096),
            ConvertFrom2DTo3D(),
            Transpose3D1x1Block(2048, 2048),

        )

        self.dec_layer5 = DecoderLayer(2048, 1024)
        self.dec_layer4 = DecoderLayer(1024, 512)
        self.dec_layer3 = DecoderLayer(512, 256)
        self.dec_layer2 = DecoderLayer(256, 128)
        self.dec_layer1 = DecoderLayer(128, 64)
        self.dec_layer0 = DecoderLayer(64, 16)

        self.output_layer = OutputBlock(16, 1)

        # 潜在空间的维度，即，我们试图将输入数据压缩到这个维度的潜在向量中
        # 如果设置的比较简单，那么训练块，但是可能存在信息丢失
        # 如果设置的比较大，那么信息比较全，但是有过拟合、训练慢等风险
        potential_channels = 100  # 20

        self.fc_mu = nn.Linear(4096 * 2 * 2, potential_channels)
        self.fc_logvar = nn.Linear(4096 * 2 * 2, potential_channels)
        self.fc_decode = nn.Linear(potential_channels, 4096 * 2 * 2)

        initialize_weights(self)

    def encode(self, data):
        pre_enc_layer_out0, enc0 = self.enc_layer0(data)

        pre_enc_layer_out1, enc1 = self.enc_layer1(enc0 + pre_enc_layer_out0)

        pre_enc_layer_out2, enc2 = self.enc_layer2(enc1 + pre_enc_layer_out1)

        pre_enc_layer_out3, enc3 = self.enc_layer3(enc2 + pre_enc_layer_out2)

        pre_enc_layer_out4, enc4 = self.enc_layer4(enc3 + pre_enc_layer_out3)

        pre_enc_layer_out5, enc5 = self.enc_layer5(enc4)

        # 平铺
        h = torch.flatten(enc5, start_dim=1)

        # 映射到潜在空间中，这个潜在的空间的平均值是mu，对数方差是logvar
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        进行重参数化
        :param mu: 平均值
        :param logvar: 对数方差
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = F.relu(h)
        data = h.view(-1, 4096, 2, 2)

        features = self.transformer_layer(data)

        dec5 = self.dec_layer5(features)
        dec4 = self.dec_layer4(dec5)
        dec3 = self.dec_layer3(dec4)

        dec2 = self.dec_layer2(dec3)
        dec1 = self.dec_layer1(dec2)
        dec0 = self.dec_layer0(dec1)

        output = self.output_layer(dec0)

        output = torch.squeeze(output, 1)

        return output

    def forward(self, x):
        # 编码器网络，映射到隐藏空间
        mu, logvar = self.encode(x)
        # 重参数化
        z = self.reparameterize(mu, logvar)
        # 根据重参数化的输出进行解码
        return self.decode(z), mu, logvar


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        pass

    def forward(self, recon_x, x, mu, logvar):
        """
        计算损失，包括两个部分，一个是重构损失，另一个是KL
        原始的重构损失是BCELoss，这里改成MSE
        :param recon_x:重建数据
        :param x:原始数据
        :param mu:均值
        :param logvar:对数方差
        :return: loss
        """

        # Reconstruction + KL divergence losses summed over all elements and batch

        """
        # 原始逻辑
        # https://arxiv.org/abs/1312.6114
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
        """

        mse = self.mse_loss(recon_x, x)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return mse + KLD


def save_ct(output_tensor):
    output_tensor = torch.squeeze(output_tensor, 1)

    numpy_output = output_tensor.data.cpu().numpy()
    nii_numpy_output = numpy_output[0] / 255.0
    nii_numpy_output = un_normalization(nii_numpy_output, CT_MIN_MAX[0], CT_MIN_MAX[1])

    out_ct = sitk.GetImageFromArray(nii_numpy_output)

    sitk.WriteImage(out_ct, "dataset/output.nii.gz")


class V6:
    def __init__(self, input_channels):
        self.best_loss = 1e5
        self.output_path = "./dataset"
        self.resume = "final"
        self.device = torch.device("cpu")
        self.model = VAE(input_channels)
        self.model = nn.DataParallel(self.model).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = VAELoss()
        self.criterion_scale_factor = 1.0  # 255.0  # 1.0

        pass

    def train(self, train_loader, epoch):
        self.model.train()

        train_loss = 0

        for batch_idx, (x_datas, ct) in enumerate(train_loader):
            ct = torch.unsqueeze(ct, 1)

            x_datas = x_datas.to(self.device)
            ct = ct.to(self.device)

            self.optimizer.zero_grad()

            recon_batch, mu, logvar = self.model(x_datas)

            recon_batch = torch.unsqueeze(recon_batch, 1)

            loss = self.criterion(recon_batch / self.criterion_scale_factor, ct / self.criterion_scale_factor, mu, logvar)

            loss.backward()

            train_loss += loss.item()

            self.optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x_datas), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           loss.item() / len(x_datas)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(train_loader.dataset)))
        pass

    def validate(self, test_loader, epoch):
        self.model.eval()

        test_loss = 0

        last_output = None

        with torch.no_grad():
            for i, (x_datas, ct) in enumerate(test_loader):

                ct = torch.unsqueeze(ct, 1)

                x_datas = x_datas.to(self.device)

                ct = ct.to(self.device)

                recon_batch, mu, logvar = self.model(x_datas)

                recon_batch = torch.unsqueeze(recon_batch, 1)

                test_loss += self.criterion(recon_batch / self.criterion_scale_factor, ct / self.criterion_scale_factor, mu, logvar).item()

                last_output = recon_batch

                """if i == 0:
                    n = min(x_datas.size(0), 8)
                    comparison = torch.cat([x_datas[:n],
                                            recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                    save_image(comparison.cpu(),
                               'results/reconstruction_' + str(epoch) + '.png', nrow=n)"""

        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        return last_output, test_loss

    def save(self, epoch, curr_val_loss):
        is_best = curr_val_loss < self.best_loss

        self.best_loss = min(curr_val_loss, self.best_loss)

        state = {'epoch': epoch + 1,
                 'state_dict': self.model.state_dict(),
                 'best_loss': self.best_loss,
                 'optimizer': self.optimizer.state_dict(),
                 }

        filename = osp.join(self.output_path, 'vae_curr_model.pth.tar')
        best_filename = osp.join(self.output_path, 'vae_best_model.pth.tar')

        print('! Saving checkpoint: {}'.format(filename))
        torch.save(state, filename)

        if is_best:
            print('!! Saving best checkpoint: {}'.format(best_filename))
            shutil.copyfile(filename, best_filename)

        pass

    def load(self):

        ckpt_file = ""

        if self.resume == 'best':
            ckpt_file = osp.join(self.output_path, 'vae_best_model.pth.tar')
        elif self.resume == 'final':
            ckpt_file = osp.join(self.output_path, 'vae_curr_model.pth.tar')
        else:
            assert False, print("=> no available checkpoint '{}'".format(ckpt_file))

        start_epoch = 0

        if not os.path.exists(ckpt_file):
            return start_epoch

        if osp.isfile(ckpt_file):
            print("=> loading checkpoint '{}'".format(ckpt_file))
            checkpoint = torch.load(ckpt_file)
            start_epoch = checkpoint['epoch']

            self.best_loss = checkpoint['best_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_file))

        return start_epoch

        pass


if __name__ == "__main__":
    net = V6(3)

    start_epoch = net.load()

    for epoch in range(start_epoch, args.epochs + 1):
        net.train(train_loader, epoch)
        last_output = None
        validate_loss = 0.0

        if (epoch + 1) % 5 == 0:
            last_output, validate_loss = net.validate(test_loader, epoch)
            save_ct(last_output)
            pass

        if (epoch + 1) % 20 == 0:
            net.save(epoch, validate_loss)
            pass
        """with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')"""
