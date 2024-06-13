from __future__ import print_function
import argparse
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from data_reader import *
from conv_block import *
from v4_resnet_unet import V4

parser = argparse.ArgumentParser()

parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--num_generator_features', type=int, default=64)
parser.add_argument('--num_discriminator_features', type=int, default=64)
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--num_gpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--enable_norm', type=bool, default=False, help='enable_norm')
parser.add_argument('--enable_adaptive_lr', type=bool, default=True, help='enable_adaptive_lr')
parser.add_argument('--train_file', type=str, default="./dataset/train.csv", help='weight_decay')
parser.add_argument('--test_file', type=str, default="./dataset/test.csv", help='weight_decay')
parser.add_argument('--data_root', type=str,
                    default="/Users/mobinsheng/work/other/pytorch-projects/PatRecon/dataset/data", help='data_root')
parser.add_argument('--output_path', type=str, default="./dataset", help='output_path')
parser.add_argument('--resume', type=str, default="final", help='resume')
parser.add_argument('--num-views', type=int, default=3, help='number of views/projections in inputs')
parser.add_argument('--input-size', type=int, default=128, help='dimension of input view size')
parser.add_argument('--output-size', type=int, default=128, help='dimension of ouput 3D model size')
parser.add_argument('--output-channel', type=int, default=128, help='dimension of ouput 3D model size')
parser.add_argument('--cgan', type=bool, default=False, help='cgan')


class Generator(nn.Module):
    """
    给生成器输入噪声，对于某些任务是需要的，例如生成一个人脸，噪声可以增加多样性
    但是对于我们当前的任务来说，不需要，因为我们是要根据实际的X光生成CT，不需要多样性
    """

    def __init__(self, num_gpu, input_channels, output_channels):
        super(Generator, self).__init__()

        self.enable_renet = False

        self.num_gpu = num_gpu

        # 生成器的网络使用的是V4
        self.model = V4(input_channels, output_channels)

    def forward(self, input):
        if input.is_cuda and self.num_gpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(self.num_gpu))
        else:
            output = self.model(input)
        return output

    pass


class Discriminator(nn.Module):
    def __init__(self, num_gpu, cgan):
        super(Discriminator, self).__init__()

        self.num_gpu = num_gpu

        """
        是否条件式gan，即数据不是Generator随机生成的，而是根据指定的条件（输入）进行生成
        对于当前的场景来说，条件就是X光片
        
        CT的维度是    [batch_size, 1, 128, 128, 128]
        X片的维度是   [batch_size, 1, 128, 128]
        需要把X和CT按照channel维度拼接起来
        """
        self.cgan = cgan

        ct_channels = 1

        if self.cgan:
            x_channels = 3
        else:
            x_channels = 0

        self.model = nn.Sequential(

            nn.Conv3d(in_channels=ct_channels + x_channels, out_channels=32, kernel_size=4, stride=2, padding=1,
                      bias=False),
            # 64x64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            # 32x32
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            # 16x16
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            # 8x8
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias=False),
            # 4x4
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def combine_data(self, ct, x_datas):
        """
        把X片的维度从[batch_size, 3, 128, 128]转换为[batch_size, 3, 128, 128, 128]
        然后和CT拼接起来，X光的通道数是3是因为它有3个X光片
        :param ct:
        :param x_datas:
        :return:
        """
        if self.cgan:
            _, channels, _, _ = x_datas.shape

            output = ct

            for c in range(channels):
                # 取出每一个X光片
                x = x_datas[:, c, :, :]
                # [batch_size, 1, 128, 128] --> [batch_size, 1, 128, 128, 128]
                x = x.unsqueeze(1).expand_as(ct)
                # 拼接
                output = torch.cat((output, x), dim=1)
                pass

            return output
        else:
            return ct

    def forward(self, CT, x_datas):

        input = self.combine_data(CT, x_datas)

        if input.is_cuda and self.num_gpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(self.num_gpu))
        else:
            output = self.model(input)

        return output.view(-1, 1).squeeze(1)

    pass


args = parser.parse_args()
print(args)

try:
    os.makedirs(args.output_path)
except OSError:
    pass

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

generator_output_channels = args.output_channel

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

device = torch.device("cpu")

num_gpu = int(args.num_gpu)
nz = int(args.nz)
num_generator_features = int(args.num_generator_features)
num_discriminator_features = int(args.num_discriminator_features)

# custom weights initialization called on netG and netD

init_network = False

netG_file = osp.join(args.output_path, 'netG.pth.tar')
netD_file = osp.join(args.output_path, 'netD.pth.tar')

netG = Generator(num_gpu, 3, 128).to(device)
if init_network:
    netG.apply(initialize_weights)

if osp.isfile(netG_file):
    netG = torch.load(netG_file)

print(netG)

netD = Discriminator(num_gpu, args.cgan).to(device)
if init_network:
    netD.apply(initialize_weights)

if osp.isfile(netD_file):
    netD = torch.load(netD_file)

print(netD)

bce_criterion = nn.BCELoss()
mse_criterion = nn.MSELoss()

fixed_noise = torch.randn(args.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

start_epoch = 0
epochs = 10000

netG.train()
netD.train()

for epoch in range(start_epoch, epochs, 1):
    for i, (x_datas, ct) in enumerate(train_loader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        ct = ct.to(device)

        # 增加一个channel维度
        ct = torch.unsqueeze(ct, 1)
        x_datas = x_datas.to(device)
        batch_size = ct.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=ct.dtype, device=device)

        netD.zero_grad()

        output = netD(ct, x_datas)
        errD_real = bce_criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(x_datas)
        # 增加一个channel维度
        fake = torch.unsqueeze(fake, 1)
        label.fill_(fake_label)
        output = netD(fake.detach(), x_datas)
        errD_fake = bce_criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake, x_datas)
        errG = bce_criterion(output, label)
        mse = mse_criterion(fake, ct)
        g_loss = mse + errG
        g_loss.backward()
        # errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Mse: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, args.niter, i, len(train_loader),
                 mse.item(), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        """
        if i % 100 == 0:
            vutils.save_image(x_datas,
                              '%s/real_samples.png' % args.output_path,
                              normalize=True)
            fake = netG(x_datas) #netG(fixed_noise)
            vutils.save_image(fake.detach(),
                              '%s/fake_samples_epoch_%03d.png' % (args.output_path, epoch),
                              normalize=True)
        """

        if args.dry_run:
            break
    # do checkpointing
    if (epoch + 1) % 20 == 0:
        torch.save(netG, '%s/netG.pth.tar' % args.output_path)
        torch.save(netD, '%s/netD.pth.tar' % args.output_path)
        pass

    if (epoch + 1) % 5 == 0:
        netG.eval()
        netD.eval()

        output = netG(x_datas)
        output = torch.squeeze(output, 1)

        numpy_output = output.data.cpu().numpy()
        nii_numpy_output = numpy_output[0] / 255.0
        nii_numpy_output = un_normalization(nii_numpy_output, CT_MIN_MAX[0], CT_MIN_MAX[1])

        out_ct = sitk.GetImageFromArray(nii_numpy_output)

        sitk.WriteImage(out_ct, "dataset/output.nii.gz")

        netD.train()
        netG.train()
