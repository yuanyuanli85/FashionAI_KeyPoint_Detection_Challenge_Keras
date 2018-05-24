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

BEST_GUESS_SKIRT = [(176, 43, 1), (327, 34, 1), (112, 459, 1), (364, 470, 1)]


def get_kp_distribution(annFile, category):
    xdf = pd.read_csv(annFile)
    xdf = xdf[xdf['image_category'] == category]

    #xdf = xdf[:1000]

    xdict = dict()
    for _index, _row in xdf.iterrows():
        imgId = _row['image_id']
        nkplst = get_normized_kp(_row, category)
        xdict[imgId] = nkplst

    heatmap = np.zeros( (512, 512, getKpNum(category)), dtype=np.float)
    for key, value in xdict.items():
        for i, _kp in enumerate(value):
            #heatmap[_kp.y, _kp.x, i] += 1
            heatmap[_kp.y-5 : _kp.y+5,
                    _kp.x-5 : _kp.x+5 , i] += 0.1

    #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    mGoldenKp = list()
    for i in range(heatmap.shape[-1]):
        _map = heatmap[:,:,i]
        #_map = cv2.normalize(_map, 0, 255, cv2.NORM_L1)
        maxlocations = np.where(_map == _map.max())
        maxY = sum(maxlocations[0])/len(maxlocations[0])
        maxX = sum(maxlocations[1])/len(maxlocations[1])
        mGoldenKp.append((int(maxX), int(maxY), 1))
        #_colormap = cv2.applyColorMap(_map, cv2.COLORMAP_JET)
        #cv2.imshow('heatmap', _map)
        #cv2.waitKey()

    print mGoldenKp, category

    return mGoldenKp


def get_normized_kp(dfrow, category):
    mlist = dfrow[getKpKeys(category)]
    imgName, kpStr = mlist[0], mlist[1:]

    cvmat = cv2.imread(os.path.join("../../data/train", imgName))
    imgHeight, imgWidth, _c = cvmat.shape

    widthScale  = imgWidth/512.0
    heightScale = imgHeight/512.0

    # read kp annotation from csv file
    kpAnnlst = [KpAnno.readFromStr(_kpstr) for _kpstr in kpStr]
    normalizedKpAnnlst =  list()
    for _kp in kpAnnlst:
        _nkp = KpAnno( int(_kp.x/widthScale), int(_kp.y/heightScale), _kp.visibility)
        normalizedKpAnnlst.append(_nkp)

    return normalizedKpAnnlst


def main():
    annfile = os.path.join("../../data/train/Annotations", "train_split.csv")
    xdict = dict()
    for category in ['skirt', 'blouse', 'dress', 'trousers', 'outwear']:
        xdict[category] = get_kp_distribution( annfile, category)
    with open('best_guess.pkl', 'w+') as xfile:
        pickle.dump(xdict, xfile)


def get_best_guess(category, partindex):
    pklfile = 'best_guess.pkl'
    with open(pklfile) as xfile:
        xdict = pickle.load(xfile)

    xpartlst = xdict[category]
    part = xpartlst[partindex]
    return KpAnno(part[0], part[1], 1)

if __name__ == "__main__":
    mkp = get_best_guess('skirt', 0)
    print mkp.x , mkp.y , mkp.visibility