
import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../unet/")
sys.path.insert(0, "../visual/")

import pandas as pd
from dataset import getKpKeys, getKpNum, getFlipMapID, generate_input_mask
from kpAnno import KpAnno
from post_process import post_process_heatmap
from keras.models import load_model
import os
from vgg_unet import euclidean_loss
import numpy as np
from best_guess import get_best_guess
import cv2
from resnet101 import Scale

class Evaluation(object):
    def __init__(self, category, modelFile):
        self.category = category
        self.train_img_path = "../../data/train"
        if modelFile is not None:
            self._initialize(modelFile)

    def init_from_model(self, model):
        self._load_anno()
        self.net = model

    def eval(self, multiOut=False, debug=False):
        xdf = self.annDataFrame
        scores = list()
        xdict = dict()
        for _index, _row in xdf.iterrows():
            imgId = _row['image_id']
            imgFile = os.path.join(self.train_img_path, imgId)
            gtKpAnno = self._get_groundtruth_kpAnno(_row)
            predKpAnno = self.predict_kp(imgFile, _row['image_category'], multiOut)
            neScore = Evaluation.calc_ne_score(self.category, predKpAnno, gtKpAnno)
            scores.extend(neScore)
            if debug:
                xdict[imgId] = (gtKpAnno, predKpAnno, neScore)

        if debug:
            return scores, xdict
        else:
            return sum(scores)/len(scores)

    def eval_mask(self):
        xdf = self.annDataFrame
        scores = list()
        xdict = dict()
        for _index, _row in xdf.iterrows():
            imgId = _row['image_id']
            imgFile = os.path.join(self.train_img_path, imgId)
            gtKpAnno = self._get_groundtruth_kpAnno(_row)
            predKpAnno = self.predict_kp_with_mask(imgFile, True)
            neScore = self.calc_ne_score(self.category, predKpAnno, gtKpAnno)
            xdict[imgId] = (gtKpAnno, predKpAnno, neScore)
            scores.extend(neScore)
        return scores, xdict

    def eval_with_flip(self):
        xdf = self.annDataFrame
        scores = list()
        xdict = dict()
        for _index, _row in xdf.iterrows():
            imgId = _row['image_id']
            imgFile = os.path.join(self.train_img_path, imgId)
            gtKpAnno = self._get_groundtruth_kpAnno(_row)
            predKpAnno = self.predict_kp_with_flip(imgFile)
            neScore = self.calc_ne_score(self.category, predKpAnno, gtKpAnno)
            xdict[imgId] = (gtKpAnno, predKpAnno, neScore)
            scores.extend(neScore)
        return scores, xdict

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
        xpd = xpd[xpd['image_category'] == self.category]
        self.annDataFrame = xpd


    def _get_groundtruth_kpAnno(self, dfrow):
        mlist = dfrow[getKpKeys(self.category)]
        imgName, kpStr = mlist[0], mlist[1:]
        # read kp annotation from csv file
        kpAnnlst = [KpAnno.readFromStr(_kpstr) for _kpstr in kpStr]
        return kpAnnlst

    def _net_inference(self, imgFile, flip=False):
        import cv2
        from data_process import normalize_image, pad_image_inference
        # load image and preprocess
        img = cv2.imread(imgFile)

        if flip:
            img = cv2.flip(img, flipCode=1)

        # flip must happen before padding due to location change
        img, scale = pad_image_inference(img, 512, 512)

        img = normalize_image(img)
        input = img[np.newaxis,:,:,:]

        # inference
        heatmap = self.net.predict(input)

        return (heatmap, scale)

    def _net_inference_with_mask(self, imgFile, imgCategory):
        import cv2
        from data_process import normalize_image, pad_image_inference
        assert (len(self.net.input_layers) == 2), "input layer need to be 2"

        # load image and preprocess
        img = cv2.imread(imgFile)

        img, scale = pad_image_inference(img, 512, 512)
        img = normalize_image(img)
        input_img = img[np.newaxis, :, :, :]

        input_mask = generate_input_mask(imgCategory, (512, 512, getKpNum('all') + 1))
        input_mask = input_mask[np.newaxis, :, :, :]

        # inference
        heatmap = self.net.predict([input_img, input_mask])

        return (heatmap, scale)

    def predict_kp_with_mask(self, imgFile, multiOutput=False):
        xnetout, scale = self._net_inference(imgFile)
        if multiOutput:
            # todo: fixme, it is tricky that the previous stage has beeter performance than last stage's output.
            # todo: here, we are using multiple stage's output sum.
            heatmap = self._heatmap_sum(xnetout)
        else:
            heatmap = xnetout

        heatmap = self.merge_with_mask(self.category, heatmap)

        detectedKps = post_process_heatmap(heatmap, kpConfidenceTh=0.2)

        # scale to padded resolution 256X256 -> 512X512
        scaleTo512 = 2.0

        # apply scale to original resolution
        detectedKps = [KpAnno(_kp.x * scaleTo512 / scale, _kp.y * scaleTo512 / scale, _kp.visibility) for _kp in
                       detectedKps]

        return detectedKps

    def merge_with_mask(self, category, heatmap):
        from kp_distribution import get_distribution_mask, get_distribution_normalized_mask

        #mask = get_distribution_mask(category)

        mask = get_distribution_normalized_mask(category)

        assert ((heatmap.shape[1], heatmap.shape[2], heatmap.shape[3]-1) == mask.shape), "mask = heatmap.shape -1"

        for i in range(mask.shape[-1]):
            heatmap[0, :,:,i] = heatmap[0, :,:,i] * mask[:,:,i]
        return heatmap

    def _heatmap_sum(self, heatmaplst):
        outheatmap = heatmaplst[0]
        for i in range(1, len(heatmaplst), 1):
            outheatmap += heatmaplst[i]
        return outheatmap

    def predict_kp(self, imgFile, imgCategory, multiOutput=False):

        if len(self.net.input_layers) == 2:
            xnetout, scale = self._net_inference_with_mask(imgFile, imgCategory)
        else:
            xnetout, scale = self._net_inference(imgFile)

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

        if False:
            #use best guess to replace kp with low confidence
            for i, _kp in enumerate(detectedKps):
                if _kp.visibility == -1:
                    normalizedKp = get_best_guess(self.category, i)
                    detectedKps[i] = KpAnno( int(normalizedKp.x / scale),
                                             int(normalizedKp.y / scale), 1)
        return detectedKps


    def predict_kp_with_flip(self, imgFile):
        # orignial inference
        xnetout, scale = self._net_inference(imgFile)
        xnetHeatmap = self._heatmap_sum(xnetout)

        xnetflipout, scale = self._net_inference(imgFile, flip=True)
        xnetFlipHeatmap = self._flip_out_heatmap(xnetflipout)

        #todo: fixme: hard code
        heatmap = xnetHeatmap + xnetFlipHeatmap

        detectedKps = post_process_heatmap(heatmap, kpConfidenceTh=0.2)

        # scale to padded resolution 256X256 -> 512X512
        scaleTo512 = 2.0

        # apply scale to original resolution
        detectedKps = [KpAnno(_kp.x * scaleTo512 / scale, _kp.y * scaleTo512 / scale, _kp.visibility) for _kp in
                       detectedKps]

        return detectedKps


    def _flip_out_heatmap(self, flipout):
        heatmap = self._heatmap_sum(flipout)
        outmap = np.zeros(heatmap.shape, dtype=np.float)
        for i in range(heatmap.shape[-1] - 1):
            flipid = getFlipMapID(self.category, i)
            mask = np.copy(heatmap[0, :, :, i])
            outmap[0, :, :, flipid] = cv2.flip(mask, flipCode=1)
        return outmap

    @staticmethod
    def get_normized_distance(category, gtKp):
        '''

        :param category:
        :param gtKp:
        :return: if ground truth's two points do not exist, return a big number 1e6
        '''

        assert (len(gtKp) == getKpNum(category)), category + " must have '" + str(getKpNum(category)) + " keypoints'"
        if category in ['skirt' ,'trousers']:
            ##waistband left and right
            if gtKp[0].visibility != -1 and gtKp[1].visibility != -1:
                distance = KpAnno.calcDistance(gtKp[0], gtKp[1])
            else:
                distance = 1e6
            return distance
        elif category in ['blouse', 'dress']:
            ##armpit_left armpit_right'
            if gtKp[5].visibility != -1 and gtKp[6].visibility != -1:
                distance = KpAnno.calcDistance(gtKp[5], gtKp[6])
            else:
                distance = 1e6
            return distance
        elif category == 'outwear':
            ##armpit_left armpit_right'
            if gtKp[4].visibility != -1 and gtKp[5].visibility != -1:
                distance = KpAnno.calcDistance(gtKp[4], gtKp[5])
            else:
                distance = 1e6 # large number
            return distance
        else:
            assert (0), "not implemented in _get_normized_distance"

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
