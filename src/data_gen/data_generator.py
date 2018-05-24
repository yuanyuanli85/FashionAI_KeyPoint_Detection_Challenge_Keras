
import os
import cv2
import pandas as pd
import numpy as np
import random

from kpAnno import KpAnno
from dataset import getKpNum, getKpKeys, getFlipMapID,  generate_input_mask
from utils import make_gaussian, load_annotation_from_df
from data_process import pad_image, resize_image, normalize_image, rotate_image, \
    rotate_image_float, rotate_mask, crop_image
from ohem import generate_topk_mask_ohem

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

    def generator_with_mask_ohem(self, graph, kerasModel, batchSize=16, inputSize=(512, 512), flipFlag=False, cropFlag=False,
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
        train_mask = np.zeros((batchSize, targetHeight / 2, targetWidth / 2, getKpNum(self.category) ), dtype=np.float)
        train_gthmap = np.zeros((batchSize, targetHeight / 2, targetWidth / 2, getKpNum(self.category) ), dtype=np.float)
        train_ohem_mask = np.zeros((batchSize, targetHeight / 2, targetWidth / 2, getKpNum(self.category) ), dtype=np.float)
        train_ohem_gthmap = np.zeros((batchSize, targetHeight / 2, targetWidth / 2, getKpNum(self.category) ), dtype=np.float)

        ## generator need to be infinite loop
        while 1:
            # random shuffle at first
            if shuffle:
                xdf = xdf.sample(frac=1)
            count = 0
            for _index, _row in xdf.iterrows():
                xindex = count % batchSize
                xinput, xhmap = self._prcoess_img(_row, inputSize, rotateFlag, flipFlag, cropFlag, nobgFlag=True)
                xmask = generate_input_mask(_row['image_category'],
                                            (targetHeight, targetWidth, getKpNum(self.category)))

                xohem_mask, xohem_gthmap = generate_topk_mask_ohem([xinput, xmask], xhmap, kerasModel, graph,
                                            8, _row['image_category'], dynamicFlag=False)

                train_input[xindex, :, :, :] = xinput
                train_mask[xindex, :, :, :] = xmask
                train_gthmap[xindex, :, :, :] = xhmap
                train_ohem_mask[xindex, :, :, :] = xohem_mask
                train_ohem_gthmap[xindex, :, :, :] = xohem_gthmap

                # if refinenet enable, refinenet has two outputs, globalnet and refinenet
                if xindex == 0 and count != 0:
                    gthamplst = list()
                    for i in range(nStackNum):
                        gthamplst.append(train_gthmap)

                    # last stack will use ohem gthmap
                    gthamplst.append(train_ohem_gthmap)

                    yield [train_input, train_mask, train_ohem_mask], gthamplst

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
        xpd = load_annotation_from_df(xpd, self.category)
        self.annDataFrame = xpd

    def _prcoess_img(self, dfrow, inputSize, rotateFlag, flipFlag, cropFlag, nobgFlag):

        mlist = dfrow[getKpKeys(self.category)]
        imgName, kpStr = mlist[0], mlist[1:]

        # read kp annotation from csv file
        kpAnnlst = list()
        for _kpstr in kpStr:
            _kpAn = KpAnno.readFromStr(_kpstr)
            kpAnnlst.append(_kpAn)

        assert (len(kpAnnlst) == getKpNum(self.category)), str(len(kpAnnlst))+" is not the same as "+str(getKpNum(self.category))


        xcvmat = cv2.imread(os.path.join(self.train_img_path, imgName))
        if xcvmat is None:
            return None, None

        #flip as first operation.
        # flip image
        if random.choice([0, 1]) and flipFlag:
            xcvmat, kpAnnlst = self.flip_image(xcvmat, kpAnnlst)

        #if cropFlag:
        #    xcvmat, kpAnnlst = crop_image(xcvmat, kpAnnlst, 0.8, 0.95)

        # pad image to 512x512
        paddedImg, kpAnnlst = pad_image(xcvmat, kpAnnlst, inputSize[0], inputSize[1])

        assert (len(kpAnnlst) == getKpNum(self.category)), str(len(kpAnnlst)) + " is not the same as " + str(
            getKpNum(self.category))

        # output ground truth heatmap is 256x256
        trainGtHmap = self.__generate_hmap(paddedImg, kpAnnlst)

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
        gthmp = np.zeros((cvmat.shape[0], cvmat.shape[1], getKpNum(self.category)), dtype=np.float)

        for i, _kpAnn in enumerate(kpAnnolst):
            if _kpAnn.visibility == -1:
                continue

            radius = 100
            gaussMask = make_gaussian(radius, radius, 20, None)

            # avoid out of boundary
            top_x, top_y = max(0, _kpAnn.x - radius/2), max(0, _kpAnn.y - radius/2)
            bottom_x, bottom_y = min(cvmat.shape[1], _kpAnn.x + radius/2), min(cvmat.shape[0], _kpAnn.y + radius/2)

            top_x_offset = top_x - (_kpAnn.x - radius/2)
            top_y_offset = top_y - (_kpAnn.y - radius/2)

            gthmp[ top_y:bottom_y, top_x:bottom_x, i] = gaussMask[top_y_offset:top_y_offset + bottom_y-top_y,
                                                                  top_x_offset:top_x_offset + bottom_x-top_x]

        return gthmp

    def flip_image(self, orgimg, orgKpAnolst):
        flipImg = cv2.flip(orgimg, flipCode=1)
        flipannlst = self.flip_annlst(orgKpAnolst, orgimg.shape)
        return flipImg, flipannlst


    def flip_annlst(self, kpannlst, imgshape):
        height, width, channels = imgshape

        # flip first
        flipAnnlst = list()
        for _kp in kpannlst:
            flip_x = width - _kp.x
            flipAnnlst.append(KpAnno(flip_x, _kp.y, _kp.visibility))

        # exchange location of flip keypoints, left->right
        outAnnlst = flipAnnlst[:]
        for i, _kp in enumerate(flipAnnlst):
            mapId = getFlipMapID('all', i)
            outAnnlst[mapId] = _kp

        return outAnnlst




