
import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../unet/")

import pandas as pd
from dataset import getKpKeys, getKpNum, getFlipMapID, get_kp_index_from_allkeys, generate_input_mask
from kpAnno import KpAnno
from post_process import post_process_heatmap
from keras.models import load_model
import os
from refinenet_mask_v3 import euclidean_loss
import numpy as np
import cv2
from resnet101 import Scale
from utils import load_annotation_from_df
from collections import defaultdict
import copy
from data_process import pad_image_inference

class Evaluation(object):
    def __init__(self, category, modelFile):
        self.category = category
        self.train_img_path = "../../data/train"
        if modelFile is not None:
            self._initialize(modelFile)

    def init_from_model(self, model):
        self._load_anno()
        self.net = model

    def eval(self, multiOut=False, details=False, flip=True):
        xdf = self.annDataFrame
        scores = list()
        xdict = dict()
        xcategoryDict = defaultdict(list)
        for _index, _row in xdf.iterrows():
            imgId = _row['image_id']
            category = _row['image_category']
            imgFile = os.path.join(self.train_img_path, imgId)
            gtKpAnno = self._get_groundtruth_kpAnno(_row)
            if flip:
                predKpAnno = self.predict_kp_with_flip(imgFile, category)
            else:
                predKpAnno = self.predict_kp(imgFile, category, multiOut)
            neScore = Evaluation.calc_ne_score(category, predKpAnno, gtKpAnno)
            scores.extend(neScore)
            if details:
                xcategoryDict[category].extend(neScore)
        if details:
            return sum(scores)/len(scores), xcategoryDict
        else:
            return sum(scores)/len(scores)

    def _initialize(self, modelFile):
        self._load_anno()
        self._initialize_network(modelFile)

    def _initialize_network(self, modelFile):
        self.net = load_model(modelFile, custom_objects={'euclidean_loss': euclidean_loss, 'Scale': Scale})

    def _load_anno(self):
        '''
        Load annotations from train.csv
        '''
        self.annfile = os.path.join("../../data/train/Annotations", "val_split.csv")

        # read into dataframe
        xpd = pd.read_csv(self.annfile)
        xpd = load_annotation_from_df(xpd, self.category)
        self.annDataFrame = xpd


    def _get_groundtruth_kpAnno(self, dfrow):
        mlist = dfrow[getKpKeys(self.category)]
        imgName, kpStr = mlist[0], mlist[1:]
        # read kp annotation from csv file
        kpAnnlst = [KpAnno.readFromStr(_kpstr) for _kpstr in kpStr]
        return kpAnnlst

    def _net_inference_with_mask(self, imgFile, imgCategory):
        import cv2
        from data_process import normalize_image, pad_image_inference
        assert (len(self.net.input_layers) > 1), "input layer need to more than 1"

        # load image and preprocess
        img = cv2.imread(imgFile)

        img, scale = pad_image_inference(img, 512, 512)
        img   = normalize_image(img)
        input_img = img[np.newaxis, :, :, :]

        input_mask = generate_input_mask(imgCategory, (512, 512, getKpNum(self.category)) )
        input_mask = input_mask[np.newaxis, :, :, :]

        # inference
        heatmap = self.net.predict([input_img, input_mask, input_mask])

        return (heatmap, scale)

    def _heatmap_sum(self, heatmaplst):
        outheatmap = np.copy(heatmaplst[0])
        for i in range(1, len(heatmaplst), 1):
            outheatmap += heatmaplst[i]
        return outheatmap

    def predict_kp(self, imgFile, imgCategory, multiOutput=False):

        xnetout, scale = self._net_inference_with_mask(imgFile, imgCategory)

        if multiOutput:
            #todo: fixme, it is tricky that the previous stage has beeter performance than last stage's output.
            #todo: here, we are using multiple stage's output sum.
            heatmap = self._heatmap_sum(xnetout)
        else:
            heatmap = xnetout

        detectedKps = post_process_heatmap(heatmap, kpConfidenceTh=0.2)

        # scale to padded resolution 256X256 -> 512X512
        scaleTo512 = 2.0

        # apply scale to original resolution
        detectedKps = [KpAnno(_kp.x*scaleTo512/scale , _kp.y*scaleTo512/scale, _kp.visibility) for _kp in detectedKps]

        return detectedKps


    def predict_kp_with_flip(self, imgFile, imgCategory):
        #  inference with flip and original image
        heatmap, scale = self._net_inference_flip(imgFile, imgCategory)

        detectedKps = post_process_heatmap(heatmap, kpConfidenceTh=0.2)

        # scale to padded resolution 256X256 -> 512X512
        scaleTo512 = 2.0

        # apply scale to original resolution
        detectedKps = [KpAnno(_kp.x * scaleTo512 / scale, _kp.y * scaleTo512 / scale, _kp.visibility) for _kp in
                       detectedKps]

        return detectedKps

    def _net_inference_flip(self, imgFile, imgCategory):
        import cv2
        from data_process import normalize_image, pad_image_inference
        assert (len(self.net.input_layers) > 1), "input layer need to more than 1"

        batch_size =2

        input_img  = np.zeros(shape=(batch_size, 512, 512, 3), dtype=np.float)
        input_mask = np.zeros(shape=(batch_size, 256, 256, getKpNum(self.category)), dtype=np.float)

        # load image and preprocess
        orgimage = cv2.imread(imgFile)

        padimg, scale = pad_image_inference(orgimage, 512, 512)
        flipimg = cv2.flip(padimg, flipCode=1)

        input_img[0,:,:,:] = normalize_image(padimg)
        input_img[1,:,:,:] = normalize_image(flipimg)

        mask = generate_input_mask(imgCategory, (512, 512, getKpNum(self.category)))
        input_mask[0,:,:,:] = mask
        input_mask[1,:,:,:] = mask

        # inference
        if len(self.net.input_layers) == 2:
            heatmap = self.net.predict([input_img, input_mask])
        elif len(self.net.input_layers) == 3:
            heatmap = self.net.predict([input_img, input_mask, input_mask])
        else:
            assert (0), str(len(self.net.input_layers)) + " should be 2 or 3 "

        # sum heatmap
        avgheatmap = self._heatmap_sum(heatmap)

        orgheatmap = avgheatmap[0,:,:,:]

        # convert to same sequency with original heatmap
        flipheatmap = avgheatmap[1,:,:,:]
        flipheatmap = self._flip_out_heatmap(flipheatmap)

        # average original and flip heatmap
        outheatmap = flipheatmap + orgheatmap
        outheatmap = outheatmap[np.newaxis, :, :, :]

        return (outheatmap, scale)

    def predict_kp_with_rotate(self, imgFile, imgCategory):
        #  inference with rotated image
        rotateheatmap = self._net_inference_rotate(imgFile, imgCategory)
        rotateheatmap = rotateheatmap[np.newaxis, :, :, :]

        # original image and flip image
        orgflipmap, scale = self._net_inference_flip(imgFile, imgCategory)
        mflipmap = cv2.resize(orgflipmap[0,:,:,:], None, fx=2.0/scale, fy=2.0/scale)

        # add mflipmap and rotateheatmap
        avgheatmap = mflipmap[np.newaxis, :, :, :]

        b, h, w , c = rotateheatmap.shape
        avgheatmap[:, 0:h, 0:w,:] += rotateheatmap

        # generate key point locations
        detectedKps = post_process_heatmap(avgheatmap, kpConfidenceTh=0.2)

        return detectedKps

    def _net_inference_rotate(self, imgFile, imgCategory):
        from data_process import normalize_image, pad_image_inference, rotate_image_with_invrmat

        # load image and preprocess
        orgimage = cv2.imread(imgFile)

        anglelst = [-20, -10, 10, 20]

        input_img  = np.zeros(shape=(len(anglelst), 512, 512, 3), dtype=np.float)
        input_mask = np.zeros(shape=(len(anglelst), 256, 256, getKpNum(self.category)), dtype=np.float)

        mlist = list()
        for i, angle in enumerate(anglelst):
            rotateimg, invRotMatrix, orgImgSize = rotate_image_with_invrmat(orgimage, angle)
            padimg, scale = pad_image_inference(rotateimg, 512, 512)
            _img = normalize_image(padimg)
            input_img[i, :, :, :] = _img
            mlist.append((scale, invRotMatrix))

        mask = generate_input_mask(imgCategory, (512, 512, getKpNum(self.category)))
        for i, angle in enumerate(anglelst):
            input_mask[i, :,:,:] = mask

        # inference
        heatmap = self.net.predict([input_img, input_mask, input_mask])
        heatmap = self._heatmap_sum(heatmap)

        # rotate back to original resolution
        sumheatmap =  np.zeros(shape=(orgimage.shape[0], orgimage.shape[1], getKpNum(self.category)), dtype=np.float)
        for i, item in enumerate(mlist):
            _heatmap = heatmap[i, :, :, :]
            _scale, _invRotMatrix = item
            _heatmap = cv2.resize(_heatmap, None, fx=2.0 / _scale, fy=2.0 / _scale)
            _invheatmap = cv2.warpAffine(_heatmap, _invRotMatrix, (orgimage.shape[1], orgimage.shape[0]))
            sumheatmap += _invheatmap

        return sumheatmap

    def _flip_out_heatmap(self, flipout):
        outmap = np.zeros(flipout.shape, dtype=np.float)
        for i in range(flipout.shape[-1]):
            flipid = getFlipMapID(self.category, i)
            mask = np.copy(flipout[:, :, i])
            outmap[:, :, flipid] = cv2.flip(mask, flipCode=1)
        return outmap


    @staticmethod
    def get_normized_distance(category, gtKp):
        '''

        :param category:
        :param gtKp:
        :return: if ground truth's two points do not exist, return a big number 1e6
        '''

        if category in ['skirt' ,'trousers']:
            ##waistband left and right
            waistband_left_index  = get_kp_index_from_allkeys('waistband_left')
            waistband_right_index = get_kp_index_from_allkeys('waistband_right')

            if gtKp[waistband_left_index].visibility != -1 and gtKp[waistband_right_index].visibility != -1:
                distance = KpAnno.calcDistance(gtKp[waistband_left_index], gtKp[waistband_right_index])
            else:
                distance = 1e6
            return distance
        elif category in ['blouse', 'dress', 'outwear']:
            armpit_left_index  = get_kp_index_from_allkeys('armpit_left')
            armpit_right_index = get_kp_index_from_allkeys('armpit_right')
            ##armpit_left armpit_right'
            if gtKp[armpit_left_index].visibility != -1 and gtKp[armpit_right_index].visibility != -1:
                distance = KpAnno.calcDistance(gtKp[armpit_left_index], gtKp[armpit_right_index])
            else:
                distance = 1e6
            return distance
        else:
            assert (0), category + " not implemented in _get_normized_distance"


    @staticmethod
    def calc_ne_score(category, dtKp, gtKp):

        assert (len(dtKp) == len(gtKp)), "predicted keypoint number should be the same as ground truth keypoints" + \
                                         str(dtKp) + " vs " + str(gtKp)

        # calculate normalized error as score
        normalizedDistance = Evaluation.get_normized_distance(category, gtKp)

        mlist = list()
        for i in range(len(gtKp)):
            if gtKp[i].visibility == 1:
                dk = KpAnno.calcDistance(dtKp[i], gtKp[i])
                mlist.append( dk/normalizedDistance)

        return mlist
