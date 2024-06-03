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
from data_process import *
import h5py

#CT_MEAN_STD = [0.0, 1.]
XRAY_MIN_MAX = [0, 255]
CT_MIN_MAX = [-5000.0, 5000.0] # 0 2500
XRAY_MEAN_STD = [0.516, 0.264] #mean=[0.516], std=[0.264] #[0.0, 1.0]
CT_CHANNEL = 128
XRAY_CHANNEL = 1


class XCTDataset(Dataset):
    """ 
    3D Reconstruction Dataset.
    训练的时候的数据集，对数据进行归一化和正态分布处理
    """

    def open(self):
        self.df = pd.read_csv(self.dataset_file)
        self.dataset_size = len(self.df)
            
            

    def process_ct(self, ct_data):
        #ct_data = ResizeAndPadimage(ct_data, self.args.input_size)
        ct_data = Resize_image(ct_data, self.args.input_size)
        ct_data = LimitThreshold(ct_data, CT_MIN_MAX[0], CT_MIN_MAX[1])
        ct_data = Normalization(ct_data, CT_MIN_MAX[0], CT_MIN_MAX[1])
        #ct_data = ct_data - np.min(ct_data)
        #ct_data = ct_data / np.max(ct_data)
        #ct_data = Standardization(ct_data, CT_MEAN_STD[0], CT_MEAN_STD[1])
        #print("CT_MAX:", np.max(ct_data))
        #print("CT_MIN:", np.min(ct_data))
        #assert((np.max(ct_data)-1.0 < 1e-3) and (np.min(ct_data) < 1e-3))
        return ct_data

    def process_x(self, x_data):
        #x_data = ResizeAndPadimage(x_data, self.args.input_size)
        x_data = Resize_image(x_data, self.args.input_size)

        x_data = Normalization(x_data, XRAY_MIN_MAX[0], XRAY_MIN_MAX[1])
        x_data = Standardization(x_data, XRAY_MEAN_STD[0], XRAY_MEAN_STD[1])
        return x_data

    def __init__(self, dataset_file, args, transform=None):
        # width = 160
        # height = 120
        # ct, depth = 160, width = 192, height = 192
        self.args = args
        self.dataset_file = dataset_file

        self.open()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        assert self.args.num_views == 1 or self.args.num_views == 3

        if self.args.file_type == "h5":
            ct_data, x_middle_data, x_left_data, x_right_data = self.read_h5(idx)
            pass
        else:
            # 读取数据，如果num_views是1，那么x_left_data和x_right_data是None
            ct_data, x_middle_data, x_left_data, x_right_data = self.read_raw(idx)

        # print(ct_data.shape, x_ray.shape)

        # 对CT数据做预处理
        ct_data = self.process_ct(ct_data)

        # 对X片做预处理
        x_middle_data = self.process_x(x_middle_data)

        x_datas = np.zeros((self.args.num_views, self.args.input_size, self.args.input_size), dtype=np.float32)

        x_datas[0, :, :] = np.array(x_middle_data)
        if self.args.num_views == 3:
            # 如果是多个X光输入，那么对剩下的X光做预处理
            x_left_data = self.process_x(x_left_data)
            x_right_data = self.process_x(x_right_data)

            x_datas[1, :, :] = np.array(x_left_data)
            x_datas[2, :, :] = np.array(x_right_data)

        ct_data = ToTensor(ct_data)
        x_datas = ToTensor(x_datas)

        return (x_datas, ct_data)

    def read_raw(self, idx):
        x_middle = self.df.iloc[idx]["x_middle"]
        ct_name = self.df.iloc[idx]["CT"]

        assert os.path.exists(x_middle), 'Path do not exist: {}'.format(x_middle)
        assert os.path.exists(ct_name), 'Path do not exist: {}'.format(ct_name)

        ct_data = read_nii_gz(ct_name)
        x_middle_data = tif_read(x_middle)

        if self.args.num_views == 3:
            x_left = self.df.iloc[idx]["x_left"]
            x_right = self.df.iloc[idx]["x_right"]
            assert os.path.exists(x_left), 'Path do not exist: {}'.format(x_left)
            assert os.path.exists(x_right), 'Path do not exist: {}'.format(x_right)

            x_left_data = tif_read(x_left)
            x_right_data = tif_read(x_right)

        return (ct_data, x_middle_data, x_left_data, x_right_data)

    def read_h5(self, idx):
        h5_path = self.df.iloc[idx]["h5_path"]

        assert os.path.exists(h5_path), 'Path do not exist: {}'.format(h5_path)

        hdf5 = h5py.File(h5_path, 'r')

        if (len(hdf5.keys()) < 3):
            print("invlaid h5:", h5_path)
            return None, None, None, None

        ct_data = np.asarray(hdf5['ct'])
        x_middle_data = np.asarray(hdf5['xray1'])

        #print(ct_data.shape)
        #print(x_middle_data.shape)

        if self.args.num_views == 3:
            x_left_data = np.asarray(hdf5['xray2'])
            x_right_data = np.asarray(hdf5['xray3'])

        return (ct_data, x_middle_data, x_left_data, x_right_data)



def PostProcess(data, name='img.nii.gz'):
    depth, height, width = data.shape
    assert depth == width and width == height

    #data = UnStandardization(data, CT_MEAN_STD[0], CT_MEAN_STD[1])
    data = UnNormalization(data, CT_MIN_MAX[0], CT_MIN_MAX[1])

    # print(data)

    img = nib.Nifti1Image(data, affine=np.eye(4))

    nib.save(img, os.path.join('save_models', name))

    return data



parser = argparse.ArgumentParser(description='PyTorch 3D Reconstruction Training')
parser.add_argument('--file_type', type=str, default="h5", help="dataset file type, raw or h5")
parser.add_argument('--train_dataset_file', type=str, default="./data/all_h5.csv", help='tran_dataset_file')
parser.add_argument('--test_dataset_file', type=str, default="./data/all_h5.csv", help='test_dataset_file')
parser.add_argument('--num-views', type=int, default=3, help='number of views/projections in inputs')
parser.add_argument('--input-size', type=int, default=128, help='dimension of input view size')
parser.add_argument('--output-size', type=int, default=128, help='dimension of ouput 3D model size')
parser.add_argument('--output-channel', type=int, default=128, help='dimension of ouput 3D model size')


if  __name__ == "__main__":
    args = parser.parse_args()
    s = XCTDataset("./data/all_h5.csv", args)
    s[0]
