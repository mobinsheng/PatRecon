import shutil
import os.path as osp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim,SSIM, MS_SSIM

class CompositeLoss(nn.Module):
    def __init__(self):
        super(CompositeLoss, self).__init__()

        #self.mse_factor = 0.4
        #self.ssim_factor = 0.3
        #self.cosin_factor = 0.3
        self.use_cosin = False

        self.mse_loss = 1.0
        self.ssim_loss = 1.0
        self.cosin_loss = 0.0

        self.mse = nn.MSELoss(size_average=True, reduce=True)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=128)
        self.cosin = F.cosine_similarity

        pass

    def forward(self, output, target):
        self.mse_loss = self.mse(output, target)
        self.ssim_loss = self.ssim(output, target)
        if self.use_cosin:
            flat_output = output.view(output.size(0), output.size(1), -1)
            flat_targget = target.view(target.size(0), target.size(1), -1)
            self.cosin_loss = self.cosin(flat_output, flat_targget, dim=2).mean()

        return (self.mse_loss + 1.0 - self.ssim_loss + 1.0 - self.cosin_loss, 
                self.mse_loss,
                self.ssim_loss,
                self.cosin_loss)
    

if __name__ == "__main__":
    m = CompositeLoss()