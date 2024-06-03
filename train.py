from trainer import *
import argparse
import os
import sys
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from net import ReconNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
#from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from nii_gz_reader import *
from tif_reader import *
from xct_dataset import *
from data_process import *
import scipy.ndimage as ndimage
from plt_utils import *

plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='PyTorch 3D Reconstruction Training')
parser.add_argument('--exp', type=int, default=1, 
                    help='experiments index')
parser.add_argument('--arch', type=str, default="ReconNet", help='arch')
parser.add_argument('--output_path', type=str, default="./save_models/", help='output_path')
parser.add_argument('--resume', type=str, default="final", help='resume')
parser.add_argument('--init_type', type=str, default="standard", help='init_type')
parser.add_argument('--init_gain', type=float, default=0.02, help='init_gain')
parser.add_argument('--lr', type=float, default=0.001, help='lr') # 0.00002
parser.add_argument('--weight_decay', type=float, default=0.2, help='weight_decay')
parser.add_argument('--loss', type=str, default="l2", help='loss')
parser.add_argument('--optim', type=str, default="adam", help='optim')
parser.add_argument('--batch_size', type=int, default=28, help='batch_size')
parser.add_argument('--file_type', type=str, default="h5", help="dataset file type, raw or h5")
parser.add_argument('--use_compose_loss', type=bool, default=False, help="use_compose_loss")
parser.add_argument('--train_dataset_file', type=str, default="./data/train.csv", help='tran_dataset_file')
parser.add_argument('--test_dataset_file', type=str, default="./data/test.csv", help='test_dataset_file')
parser.add_argument('--print_freq', type=int, default=100,
                    help='print_freq')
parser.add_argument('--num-views', type=int, default=3,
                    help='number of views/projections in inputs')
parser.add_argument('--input-size', type=int, default=128,
                    help='dimension of input view size')
parser.add_argument('--output-size', type=int, default=128,
                    help='dimension of ouput 3D model size')
parser.add_argument('--output-channel', type=int, default=128,
                    help='dimension of ouput 3D model size')
parser.add_argument('--start-slice', type=int, default=0,
                    help='the idx of start slice in 3D model')
parser.add_argument('--test', type=int, default=1,
                    help='number of total testing samples')
parser.add_argument('--vis_plane', type=int, default=0,
                    help='visualization plane of 3D images: [0,1,2]')



        
if __name__ == "__main__":

    args = parser.parse_args()

    mean = 0.5 # 0.516
    std = 0.5 # 0.264

    transforms = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[mean], std=[std]),
                        ])

    train_dataset = XCTDataset(args.train_dataset_file, args, transform = transforms)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4)
    
    test_dataset = XCTDataset(args.test_dataset_file, args, transform = transforms)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4)

    model = Trainer_ReconNet(args)

    ckpt_file = osp.join(args.output_path, 'curr_model.pth.tar')

    recover_from_break = False
    start_epoch = 0

    if osp.isfile(ckpt_file):
        recover_from_break = True
        start_epoch = model.load()

    epochs = 100

    step = 1

    h = PltHelper()

    for epoch in range(start_epoch, epochs, step):
        
        train_loss = model.train_epoch(train_dataloader, epoch)
        validation_loss, last_output = model.validate(test_dataloader)

        data = np.zeros((1, args.output_channel, args.output_size, args.output_size), dtype=np.float32)
        data[0, :, :, :] = last_output.cpu().data.float()
        PostProcess(data[0])

        h.append(epoch, train_loss, validation_loss)
        #if (epoch + 1) % 10 == 0:
        #model.save(validation_loss, epoch=epoch)

    model.save(validation_loss, epoch=epochs - 1)

    pass