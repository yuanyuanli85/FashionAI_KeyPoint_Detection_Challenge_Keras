
import sys
sys.path.insert(0, "../visual/")

import os
import cv2
import pandas as pd
import numpy as np
import random

from kpAnno import KpAnno
from dataset import getKpNum, getKpKeys, getFlipMapID, generate_input_mask
from utils import make_gaussian
from data_process import pad_image, resize_image, normalize_image, rotate_image, \
    rotate_image_float, rotate_mask, crop_image


class DataGenerator(object):

    def __init__(self, category, annfile):
        self.category = category
        self.annfile  = annfile
        self._initialize()

    def get_dim_order(self):
        # default tensorflow dim order
        return "channels_last"

    def get_dataset_size(self):
        return len(self.annDataFrame)

    def generator(self, mode, batchSize=16, inputSize=(512, 512), flatten=False, flipFlag=True, cropFlag=True,
                  shuffle=True, rotateFlag=True, refineNetFlag=False, nStackNum=1):
        '''
        Input:  batch_size * Height (512) * Width (512) * Channel (3)
        Output: batch_size * Height/2 (256) * Width/2 (256) * Channel (N+1)
        '''
        xdf = self.annDataFrame

        targetHeight, targetWidth = inputSize

        # train_input: npfloat,  height, width, channels
        # train_gthmap: npfloat, N heatmap + 1 background heatmap,
        train_input  = np.zeros((batchSize, targetHeight, targetWidth, 3), dtype=np.float)
        if flatten:
            train_gthmap = np.zeros((batchSize, (targetHeight / 2) * (targetWidth / 2), getKpNum(self.category) + 1),
                                    dtype=np.float)
        else:
            train_gthmap = np.zeros((batchSize, targetHeight/2, targetWidth/2, getKpNum(self.category)+1), dtype=np.float)

        ## generator need to be infinite loop
        while 1:
            # random shuffle at first
            if shuffle:
                xdf = xdf.sample(frac=1)
            count = 0
            for _index, _row in xdf.iterrows():
                xindex = count%batchSize
                xinput, xhmap = self._prcoess_img(_row, inputSize, rotateFlag, flipFlag, cropFlag)
                if xinput is None:
                    continue 
                train_input [xindex,  :, :, :] = xinput
                if flatten:
                    h, w, c = xhmap.shape
                    train_gthmap[xindex, :, :] = xhmap.reshape((h*w, c))
                else:
                    train_gthmap[xindex, :, :, :] = xhmap

                # if refinenet enable, refinenet has two outputs, globalnet and refinenet
                if refineNetFlag:
                    if xindex == 0 and count != 0:
                        gthamplst = list()
                        for i in range(nStackNum+1):
                            gthamplst.append(train_gthmap)
                        yield train_input, gthamplst
                else:
                    if xindex == 0 and count != 0:
                        yield train_input, train_gthmap
                count += 1

    def generator_with_mask(self, mode, batchSize=16, inputSize=(512, 512), flipFlag=True, cropFlag=True,
                            shuffle=True, rotateFlag=True, nStackNum=1):

        '''
        Input:  batch_size * Height (512) * Width (512) * Channel (3)
        Input:  batch_size * 256 * 256 * Channel (N+1). Mask for each category. 1.0 for valid parts in category. 0.0 for invalid parts
        Output: batch_size * Height/2 (256) * Width/2 (256) * Channel (N+1)
        '''
        xdf = self.annDataFrame

        targetHeight, targetWidth = inputSize

        # train_input: npfloat,  height, width, channels
        # train_gthmap: npfloat, N heatmap + 1 background heatmap,
        train_input = np.zeros((batchSize, targetHeight, targetWidth, 3), dtype=np.float)
        train_mask = np.zeros((batchSize, targetHeight / 2, targetWidth / 2, getKpNum("all") + 1), dtype=np.float)
        train_gthmap = np.zeros((batchSize, targetHeight / 2, targetWidth / 2, getKpNum(self.category) + 1), dtype=np.float)

        ## generator need to be infinite loop
        while 1:
            # random shuffle at first
            if shuffle:
                xdf = xdf.sample(frac=1)
            count = 0
            for _index, _row in xdf.iterrows():
                xindex = count % batchSize
                xinput, xhmap = self._prcoess_img(_row, inputSize, rotateFlag, flipFlag, cropFlag)
                xmask = generate_input_mask(_row['image_category'],
                                            (targetHeight, targetWidth, getKpNum('all') + 1))
                if xinput is None:
                    continue
                train_input[xindex, :, :, :] = xinput
                train_mask[xindex, :, :, :] = xmask
                train_gthmap[xindex, :, :, :] = xhmap

                # if refinenet enable, refinenet has two outputs, globalnet and refinenet
                if xindex == 0 and count != 0:
                    gthamplst = list()
                    for i in range(nStackNum + 1):
                        gthamplst.append(train_gthmap)
                    yield [train_input, train_mask], gthamplst

                count += 1

    def _initialize(self):
        self._load_anno()

    def _load_anno(self):
        '''
        Load annotations from train.csv
        '''
        # Todo: check if category legal
        self.train_img_path = "../../data/train"

        # read into dataframe
        xpd = pd.read_csv(self.annfile)
        xpd = xpd[xpd['image_category'] == self.category]
        self.annDataFrame = xpd

    def _prcoess_img(self, dfrow, inputSize, rotateFlag, flipFlag, cropFlag):

        mlist = dfrow[getKpKeys(self.category)]
        imgName, kpStr = mlist[0], mlist[1:]

        # read kp annotation from csv file
        kpAnnlst = list()
        for _kpstr in kpStr:
            kpAnnlst.append(KpAnno.readFromStr(_kpstr))

        xcvmat = cv2.imread(os.path.join(self.train_img_path, imgName))
        if xcvmat is None:
            return None, None

        if cropFlag:
            xcvmat, kpAnnlst = crop_image(xcvmat, kpAnnlst, 0.8, 0.95)

        # pad image to 512x512
        paddedImg, kpAnnlst = pad_image(xcvmat, kpAnnlst, inputSize[0], inputSize[1])

        # output ground truth heatmap is 256x256
        trainGtHmap = self.__generate_hmap(paddedImg, kpAnnlst)

        # flip image
        if random.choice([0, 1]) and flipFlag:
            paddedImg, trainGtHmap = self._flip_hmap_image(self.category, paddedImg, trainGtHmap)

        if random.choice([0,1]) and rotateFlag:
            rAngle = np.random.randint(-1*40, 40)
            rotatedImage,  _ = rotate_image(paddedImg, list(), rAngle)
            rotatedGtHmap  = rotate_mask(trainGtHmap, rAngle)
        else:
            rotatedImage  = paddedImg
            rotatedGtHmap = trainGtHmap

        # resize image
        resizedImg    = cv2.resize(rotatedImage, inputSize)
        resizedGtHmap = cv2.resize(rotatedGtHmap, (inputSize[0]//2, inputSize[1]//2))

        return normalize_image(resizedImg), resizedGtHmap


    def __generate_hmap(self, cvmat, kpAnnolst):
        # kpnum + background
        gthmp = np.zeros((cvmat.shape[0], cvmat.shape[1], getKpNum(self.category) + 1), dtype=np.float)

        for i, _kpAnn in enumerate(kpAnnolst):
            radius = 100
            gaussMask = make_gaussian(radius, radius, 20, None)

            # avoid out of boundary
            top_x, top_y = max(0, _kpAnn.x - radius/2), max(0, _kpAnn.y - radius/2)
            bottom_x, bottom_y = min(cvmat.shape[1], _kpAnn.x + radius/2), min(cvmat.shape[0], _kpAnn.y + radius/2)

            top_x_offset = top_x - (_kpAnn.x - radius/2)
            top_y_offset = top_y - (_kpAnn.y - radius/2)

            gthmp[ top_y:bottom_y, top_x:bottom_x, i] = gaussMask[top_y_offset:top_y_offset + bottom_y-top_y,
                                                                  top_x_offset:top_x_offset + bottom_x-top_x]

        # background at last channel
        max_mask= np.zeros(( cvmat.shape[0], cvmat.shape[1]), dtype=np.float)

        for i in range(getKpNum(self.category)):
            max_mask = np.maximum(max_mask, gthmp[: , :, i])
        gthmp[:, :, -1] = 1 - max_mask

        return gthmp


    def _flip_hmap_image(self, category, cvmat, gthmap):
        flipped_gthmp = np.zeros(gthmap.shape, dtype=np.float)
        flipped_cvmat = cv2.flip(cvmat, flipCode=1)

        # exclude background
        for i in range(gthmap.shape[-1]-1):
            _gtmap   = gthmap[:,:, i]
            _flipmap = cv2.flip(_gtmap, flipCode=1)
            mapPartID = getFlipMapID(category, i)
            flipped_gthmp[:,:, mapPartID] = _flipmap

        # for background
        flipped_gthmp[:, :, -1] = cv2.flip(gthmap[:,:,-1], flipCode=1)
        return flipped_cvmat, flipped_gthmp




