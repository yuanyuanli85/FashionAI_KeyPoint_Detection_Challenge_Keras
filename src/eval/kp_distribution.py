
import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../unet/")
sys.path.insert(0, "../visual/")

import os
from kpAnno import KpAnno
import time
import pickle
import cv2
import pandas as pd
from dataset import getKpKeys, getKpNum
import numpy as np
from PIL import Image


def get_kp_distribution(annFile, category):
    xdf = pd.read_csv(annFile)
    xdf = xdf[xdf['image_category'] == category]

    xdict = dict()
    for _index, _row in xdf.iterrows():
        imgId = _row['image_id']
        nkplst = get_normized_kp(_row, category)
        xdict[imgId] = nkplst

    kpDistribution = np.zeros( (256, 256, getKpNum(category)), dtype=np.float)
    for key, value in xdict.items():
        for i, _kp in enumerate(value):
            #heatmap[_kp.y, _kp.x, i] += 1
            kpDistribution[_kp.y-5 : _kp.y+5,
                           _kp.x-5 : _kp.x+5 , i] += 0.1

    kpDistribution = cv2.normalize(kpDistribution, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return kpDistribution

def get_normized_kp(dfrow, category):
    mlist = dfrow[getKpKeys(category)]
    imgName, kpStr = mlist[0], mlist[1:]

    #cvmat = cv2.imread(os.path.join("../../data/train", imgName))
    #imgHeight, imgWidth, _c = cvmat.shape

    im = Image.open(os.path.join("../../data/train", imgName))
    imgWidth, imgHeight = im.size

    widthScale  = imgWidth/256.0
    heightScale = imgHeight/256.0

    # read kp annotation from csv file
    kpAnnlst = [KpAnno.readFromStr(_kpstr) for _kpstr in kpStr]
    normalizedKpAnnlst =  list()
    for _kp in kpAnnlst:
        _nkp = KpAnno( int(_kp.x/widthScale), int(_kp.y/heightScale), _kp.visibility)
        normalizedKpAnnlst.append(_nkp)

    return normalizedKpAnnlst

def get_normized_dtkp(kpAnnlst, image_id):
    im = Image.open(os.path.join("../../data/train", image_id))
    imgWidth, imgHeight = im.size

    widthScale = imgWidth / 256.0
    heightScale = imgHeight / 256.0

    # read kp annotation from csv file
    normalizedKpAnnlst = list()
    for _kp in kpAnnlst:
        _nkp = KpAnno(int(_kp.x / widthScale), int(_kp.y / heightScale), _kp.visibility)
        normalizedKpAnnlst.append(_nkp)

    return normalizedKpAnnlst

def get_distribution_mask(category):
    with open("../../data/kp_distribution.pkl") as xfile:
        xdict = pickle.load(xfile)

    return xdict[category]

def get_distribution_normalized_mask(category):
    kpdist = get_distribution_mask(category)
    for i in range(kpdist.shape[-1]):
        xmap = kpdist[:,:,i]
        nmap = cv2.normalize(xmap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        kpdist[:, :, i] = nmap

    return kpdist

def main():
    annfile = os.path.join("../../data/train/Annotations", "train_split.csv")

    xdict = dict()
    for category in ['skirt', 'blouse', 'dress', 'trousers', 'outwear']:
        print "load category ", category
        xdict[category] = get_kp_distribution( annfile, category)

    with open('../../data/kp_distribution.pkl', 'w+') as xfile:
        pickle.dump(xdict, xfile)

def get_detected_distribution(category):
    xfile = os.path.join("../../eval_temp", category+'.pkl')
    with open(xfile) as pklfile:
        xdict = pickle.load(pklfile)

    kpDistribution = np.zeros((256, 256, getKpNum(category)), dtype=np.float)

    for key, value in xdict.items():
        gtKpAnno, predKpAnno, neScore = value
        predKpAnno = get_normized_dtkp(predKpAnno, key)
        for i, _kp in enumerate(predKpAnno):
            # heatmap[_kp.y, _kp.x, i] += 1
            kpDistribution[_kp.y - 5: _kp.y + 5,
                           _kp.x - 5: _kp.x + 5, i] += 0.1

    kpDistribution = cv2.normalize(kpDistribution, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return kpDistribution

def get_overlap_kp(kplst, threshold):
    mlist = list()
    for i in range(len(kplst)):
        xkp = kplst[i]
        for j in range(i+1, len(kplst)):
            _dis = KpAnno.calcDistance(xkp, kplst[j])
            if _dis < threshold:
                mlist.append((xkp, kplst[j]))
    return mlist

def get_overlapped_dtkp(category):
    xfile = os.path.join("../../eval_temp", category+'.pkl')
    with open(xfile) as pklfile:
        xdict = pickle.load(pklfile)

    overdict = dict()
    for image_id, value in xdict.items():
        gtKpAnno, predKpAnno, neScore = value
        overkplst = get_overlap_kp(predKpAnno, 10)
        if len(overkplst) == 0:
            continue
        overdict[image_id] = overkplst

    return overdict


def visualize_predicted_keypoint(cvmat, kpoints, color, radius=7):
    for _kpAnn in kpoints:
        cv2.circle(cvmat, (_kpAnn.x, _kpAnn.y), radius=radius, color=color, thickness=2)
    return cvmat

'''
Have performance improvement on outwear, drop on other types.
../../trained_models/blouse/2018_03_28_16_53_58/blouse_weights_23.hdf5 0.0351579070828 0.0471629887332
../../trained_models/dress/2018_03_28_16_19_59/dress_weights_25.hdf5 0.0652740429229 0.0647244288318
../../trained_models/outwear/2018_03_28_16_55_47/outwear_weights_25.hdf5 0.0492414384113 0.0354971702938
../../trained_models/skirt/2018_03_28_14_42_35/skirt_weights_14.hdf5 0.0588873056834 0.0995686221507
../../trained_models/trousers/2018_03_28_12_51_59/trousers_weights_21.hdf5 0.0629093480152 0.0949166502501
'''

def main_vis():
    category = 'skirt'
    kpdist = get_distribution_normalized_mask(category)
    dtkp   = get_detected_distribution(category)
    for i in range(kpdist.shape[-1]):
        xmap = kpdist[:,:,i]
        dtmap = dtkp[:,:, i]
        cv2.imshow('gt'+str(i), xmap)
        cv2.imshow('detect'+str(i), dtmap)
        cv2.waitKey()

def debug():
    category = 'trousers'
    overlapdict = get_overlapped_dtkp(category)
    print len(overlapdict)
    for image_id, kppairlst in overlapdict.items():
        xcvmat = cv2.imread( os.path.join("../../data/train", image_id))
        for kppair in kppairlst:
            print kppair[0].x , kppair[0].y, kppair[1].x, kppair[1].y
            visualize_predicted_keypoint(xcvmat, [kppair[0]],  (0,0,255))
            visualize_predicted_keypoint(xcvmat, [kppair[1]], (255, 0, 255), radius=9)

        cv2.imshow('x', xcvmat)
        cv2.waitKey()

if __name__ == "__main__":
    debug()