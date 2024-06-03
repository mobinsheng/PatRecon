import os
from nii_gz_reader import *
from tif_reader import *
import scipy.ndimage as ndimage
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class ListCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_list):
        for t_list in self.transforms:
            # deal with separately
            if _isArrayLike(t_list):
                if len(t_list) > 1:
                    new_img_list = []
                    for img, t in zip(img_list, t_list):
                        if t is None:
                            new_img_list.append(img)
                        else:
                            new_img_list.append(t(img))
                    img_list = new_img_list
                # deal with combined
                else:
                    img_list = t_list[0](img_list)
            else:
                if t_list is not None:
                    img_list = t_list(img_list)

        return img_list


def LimitThreshold(img, min, max):
    '''
    限制取值范围，比如灰度图，它的取值范围要在0~255之间
    Restrict in value range. value > max = max,
    value < min = min
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    img_copy = img.copy()
    img_copy[img_copy > max] = max
    img_copy[img_copy < min] = min

    return img_copy


def Normalization(img, min, max, round_v=6):
    '''
    归一化
    把取值范围规整到0~1，图像的像素值取值范围是0~255，
    To value range 0-1
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''

    range = np.array((min, max), dtype=np.float32)
    img_copy = img.copy()
    img_copy = np.round((img_copy - range[0]) / (range[1] - range[0]), round_v)
    #img_copy = np.round((img_copy - range[0]) / (range[1]), round_v)
    return img_copy


def Standardization(img, mean, std):
    '''
    标准化
    正态分布处理
    To value range 0-1
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    img_copy = img.copy()
    img_copy = (img_copy - mean) / std

    return img_copy


def UnStandardization(input_image, mean, std):
    """
    Standardization的逆操作
    """
    image = input_image * std + mean
    return image


def UnNormalization(input_image, min, max):
    """
    Normalization
    """
    image = input_image * (max - min) + min
    #image = input_image * (max) + min
    return image


def ResizeAndPadimage(img, target_size):
    '''
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    if len(img.shape) == 3:
        depth, height, width = img.shape
    else:
        depth = 1
        height, width = img.shape

    # print("raw depth, height, width", depth, height, width)

    pad_width = 0
    pad_height = 0
    pad_depth = 0

    if depth > 1:
        max_value = max(depth, height, width)
        pad_width = (max_value - width) // 2
        pad_height = (max_value - height) // 2
        pad_depth = (max_value - depth) // 2

        pad = ((pad_depth, pad_depth), (pad_height, pad_height), (pad_width, pad_width))

        pad_img = np.pad(img, pad, mode="constant", constant_values=0)

        depth, height, width = pad_img.shape

        ori_shape = np.array((depth, height, width), dtype=np.float32)
        resize_factor = (target_size, target_size, target_size) / ori_shape
        img_copy = ndimage.interpolation.zoom(pad_img, resize_factor, order=1)
    else:
        max_value = max(1, height, width)
        pad_width = (max_value - width) // 2
        pad_height = (max_value - height) // 2
        pad_depth = 0

        pad = ((pad_height, pad_height), (pad_width, pad_width))

        pad_img = np.pad(img, pad, mode="constant", constant_values=0)

        height, width = pad_img.shape

        ori_shape = np.array((height, width), dtype=np.float32)
        resize_factor = (target_size, target_size) / ori_shape
        img_copy = ndimage.interpolation.zoom(pad_img, resize_factor, order=1)

    return img_copy

def Resize_image(img, target_size):
  '''
    Returns:
      img: 3d array, (z,y,x) or (D, H, W)
  '''
  if len(img.shape) == 3:
      z, x, y = img.shape
      ori_shape = np.array((z, x, y), dtype=np.float32)
      resize_factor = (target_size, target_size, target_size) / ori_shape
  else:
      x, y = img.shape
      ori_shape = np.array((x, y), dtype=np.float32)
      resize_factor = (target_size, target_size) / ori_shape
  
  
  #scipy.ndimage.interpolation.zoom()函数用于缩放数组，即使用order顺序的样条插值来缩放数组。
  img_copy = ndimage.interpolation.zoom(img, resize_factor, order=1)
  return img_copy

def ToTensor(img):
    '''
    To Torch Tensor
    img: 3D, (z, y, x) or (D, H, W)
      Returns:
        img: 3d array, (z,y,x) or (D, H, W)
    '''
    img = torch.from_numpy(img.astype(np.float32))
    return img


def SplitDataset(input_csv, train_csv, test_csv):
    df = pd.read_csv(input_csv)
    y = df["CT"]  # .to_numpy()
    x = df[["x_middle", "x_left", "x_right"]]  # .to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)

    train_data.to_csv(train_csv, index=False)
    test_data.to_csv(test_csv, index=False)


if __name__ == "__main__":
    # /home/lwj/imageDRR/all_oral120/0258_0/middle/1_7-drr.tif
    # /home/lwj/imageDRR/all_oral120/0258_0/left/1_7-drr.tif
    # /home/lwj/imageDRR/all_oral120/0258_0/right/1_7-drr.tif
    # /home/lwj/home/lwj/oral_data/nii/all_oral/0258_0/fourCm/1_7.nii.gz

    from nii_gz_reader import *
    from tif_reader import *
    from xct_dataset import PostProcess
    XRAY_MIN_MAX = [0, 255]
    CT_MIN_MAX = [-5000.0, 5000.0] # 0 2500
    XRAY_MEAN_STD = [0.516, 0.264] #mean=[0.516], std=[0.264] #[0.0, 1.0]
    CT_CHANNEL = 128
    XRAY_CHANNEL = 1

    ct_file = "/home/lwj/home/lwj/oral_data/nii/all_oral/0258_0/fourCm/1_7.nii.gz"
    x_file = "/home/lwj/imageDRR/all_oral120/0258_0/middle/1_7-drr.tif"

    #/home/lwj/home/lwj/oral_data/nii/all_oral/0573_1/fourCm/0_3.nii.gz
    #/home/lwj/imageDRR/all_oral120/0573_1/middle/0_3-drr.tif
    #/home/lwj/imageDRR/all_oral120/0573_1/left/0_3-drr.tif
    #/home/lwj/imageDRR/all_oral120/0573_1/right/0_3-drr.tif


    ct_data = read_nii_gz(ct_file)
    ct_data = ResizeAndPadimage(ct_data, 128)
    ct_data = LimitThreshold(ct_data, CT_MIN_MAX[0], CT_MIN_MAX[1])
    ct_data = Normalization(ct_data, CT_MIN_MAX[0], CT_MIN_MAX[1])
    PostProcess(ct_data, "post.nii.gz")

    
    x_data = tif_read(x_file)
    x_data = ResizeAndPadimage(x_data, 128)
    x_data = Normalization(x_data, XRAY_MIN_MAX[0], XRAY_MIN_MAX[1])
    x_data = Standardization(x_data, XRAY_MEAN_STD[0], XRAY_MEAN_STD[1])

    x_data = UnStandardization(x_data, XRAY_MEAN_STD[0], XRAY_MEAN_STD[1])
    x_data = UnNormalization(x_data, XRAY_MIN_MAX[0], XRAY_MIN_MAX[1])
    tif_save(x_data, "./save_models/post.tif")
