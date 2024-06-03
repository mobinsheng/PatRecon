# -*- coding: UTF-8 -*-
import nibabel as nib
from nibabel import nifti1

def read_nii_gz(path):
    img = nib.load(path)
    #print(img.header['db_name'])  # headr info
    # D,H,W
    img=img.get_fdata()
    return img


if __name__ == '__main__':
    data = read_nii_gz("/home/lwj/oral_data/nii/240ori/0/0_0.nii.gz")
    print(data.shape)