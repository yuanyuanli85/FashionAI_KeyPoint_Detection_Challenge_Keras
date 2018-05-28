import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../eval/")
sys.path.insert(0, "../unet/")

import argparse
import os
from fashion_net import FashionNet
from dataset import getKpNum, getKpKeys
import pandas as pd
from evaluation import Evaluation
import pickle
import numpy as np


def get_best_single_model(valfile):
    '''
    :param valfile: the log file with validation score for each snapshot
    :return: model file and score
    '''

    def get_key(item):
        return item[1]

    with open(valfile) as xval:
        lines = xval.readlines()

    xlist = list()
    for linenum, xline in enumerate(lines):
        if 'hdf5' in xline and 'Socre' in xline:
            modelname = xline.strip().split(',')[0]
            overallscore = xline.strip().split(',')[1]
            xlist.append((modelname, overallscore))

    bestmodel = sorted(xlist, key=get_key)[0]

    return bestmodel


def fill_dataframe(kplst, keys, dfrow, image_category):
    # fill category

    dfrow['image_category'] = image_category

    assert (len(keys) == len(kplst)), str(len(kplst)) + ' must be the same as ' + str(len(keys))
    for i, _key in enumerate(keys):
        kpann = kplst[i]
        outstr = str(int(kpann.x))+"_"+str(int(kpann.y))+"_"+str(1)
        dfrow[_key] = outstr

def get_kp_from_dict(mdict, image_category, image_id):
    if image_category in mdict.keys():
        xdict = mdict[image_category]
    else:
        xdict = mdict['all']
    return xdict[image_id]

def submission(pklpath):
    xdf = pd.read_csv("../../data/train/Annotations/train.csv")
    trainKeys = xdf.keys()

    testdf = pd.read_csv("../../data/test/test.csv")
    print len(testdf), " samples in test.csv"

    mdict = dict()
    for xfile in os.listdir(pklpath):
        if xfile.endswith('.pkl'):
            category = xfile.strip().split('.')[0]
            pkl = open(os.path.join(pklpath, xfile))
            mdict[category] = pickle.load(pkl)

    print testdf.keys()
    print mdict.keys()

    submissionDf = pd.DataFrame(columns=trainKeys, index=np.arange(testdf.shape[0]))
    submissionDf = submissionDf.fillna(value='-1_-1_-1')
    submissionDf['image_id'] = testdf['image_id']
    submissionDf['image_category'] = testdf['image_category']

    for _index, _row in submissionDf.iterrows():
        image_id = _row['image_id']
        image_category = _row['image_category']
        kplst = get_kp_from_dict(mdict, image_category, image_id)
        fill_dataframe(kplst, getKpKeys('all')[1:], _row, image_category)


    print len(submissionDf), "save to ",  os.path.join(pklpath, 'submission.csv')
    submissionDf.to_csv( os.path.join(pklpath, 'submission.csv'), index=False )


def load_image_names(annfile, category):
    # read into dataframe
    xdf = pd.read_csv(annfile)
    xdf = xdf[xdf['image_category'] == category]
    return xdf

def main_test(savepath, modelpath, augmentFlag):

    valfile = os.path.join(modelpath, 'val.log')
    bestmodels = get_best_single_model(valfile)

    print bestmodels, augmentFlag

    xEval = Evaluation('all', bestmodels[0])

    # load images and run prediction
    testfile = os.path.join("../../data/test/", 'test.csv')

    for category in ['skirt', 'blouse', 'trousers', 'outwear', 'dress']:
        xdict = dict()
        xdf = load_image_names(testfile, category)
        print len(xdf), " images to process ", category

        count = 0
        for _index, _row in xdf.iterrows():
            count += 1
            if count%1000 == 0:
                print count, "images have been processed"

            _image_id = _row['image_id']
            imageName = os.path.join("../../data/test", _image_id)
            if augmentFlag:
                dtkp = xEval.predict_kp_with_rotate(imageName, _row['image_category'])
            else:
                dtkp = xEval.predict_kp(imageName, _row['image_category'], multiOutput=True)
            xdict[_image_id] = dtkp

        savefile = os.path.join(savepath, category+'.pkl')
        with open(savefile, 'wb') as xfile:
            pickle.dump(xdict, xfile)

        print "prediction save to ", savefile


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--modelpath", help="path of trained model")
    parser.add_argument("--outpath", help="path to save predicted keypoints")
    parser.add_argument("--augment", default=False, type=bool, help="augment or not")

    args = parser.parse_args()

    print args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    main_test(args.outpath, args.modelpath, args.augment)
    submission(args.outpath)