# -*- coding: UTF-8 -*-
from PIL import Image
import tifffile as tf
import numpy as np

def tif_read(path): 

    img = Image.open(path) 

    #print("width:", img.width)
    #print("height:", img.height)
    # H,W
    arr = np.array(img)

    return arr

def tif_save(img, path):
    tf.imwrite(path, img)

if __name__ == '__main__':
    data = tif_read("/home/lwj/oral_data/x-ray/zwqZ/0/middle/0_0-drr.tif")
    print(data.shape)