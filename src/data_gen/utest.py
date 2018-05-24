
from data_generator import DataGenerator
import cv2
import numpy as np
import utils
import os

def main_x():
    from dataset import getKpNum
    for category in ['skirt', 'blouse', 'outwear', 'trousers', 'dress']:
        print category, getKpNum(category)

def main_split():
    xpath = "../../data/train/Annotations"

    utils.split_csv_train_val(os.path.join(xpath, "train.csv"),
                              os.path.join(xpath, "train_split.csv"),
                              os.path.join(xpath, "val_split.csv"))

def main_vis():
    xdata = DataGenerator('trousers',
                          os.path.join( "../../data/train/Annotations", "train_split.csv"))

    print xdata.get_dim_order()
    count = 0
    for _img, _gthmap in xdata.generator('train', batchSize=16, flatten=False):
        count += 16
        if count > 10000:
            break
        print _img.shape, _gthmap.shape

        for i in range(_img.shape[0]):
            cvmat = _img[i,:,:,:] + 0.5
            mask  = _gthmap[i, :, :, :]
            print i, cvmat.shape , mask.shape
            cv2.imshow('image', cvmat)
            cv2.imshow('backgroud mask', mask[:,:, -1])
            cv2.waitKey()

    print 'scan done'


if __name__ == "__main__":
    main_vis()