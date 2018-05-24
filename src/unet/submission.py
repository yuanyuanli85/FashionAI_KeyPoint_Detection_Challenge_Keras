import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../visual/")

import argparse
import pandas as pd
import numpy as np
import os
import pickle
from dataset import fill_dataframe, getKpKeys
import cv2
from kpAnno import KpAnno

def submission(pklpath):

    xdf = pd.read_csv("../../data/train/Annotations/train.csv")
    trainKeys = xdf.keys()

    testdf = pd.read_csv("../../data/test/test.csv")
    print len(testdf), " samples in test.csv"

    outdf = pd.DataFrame(columns=trainKeys)
    for xfile in os.listdir(pklpath):
        if xfile.endswith('.pkl'):
            category = xfile.split('.')[0]
            mfile = os.path.join(pklpath, xfile)
            sdf = generate_submisson_df(category,mfile,trainKeys)
            outdf = outdf.append(sdf)

    assert (len(testdf) == len(outdf)), str(len(testdf))+" records in test.csv should be the same as submission " + str(len(outdf))

    print len(outdf), "save to ",  os.path.join(pklpath, 'submission.csv')
    outdf.to_csv( os.path.join(pklpath, 'submission.csv'), index=False )


def generate_submisson_df(category, pklfile, dfkeys):
    xpkl = pickle.load(open(pklfile))
    submissionDf = pd.DataFrame(columns=dfkeys, index=np.arange(len(xpkl)))
    submissionDf = submissionDf.fillna(value='-1_-1_-1')
    submissionDf['image_id'] = xpkl.keys()

    for _index, _row in submissionDf.iterrows():
        _imageId = _row['image_id']
        kplst = xpkl[_imageId]
        fill_dataframe(kplst, category, _row)

    return submissionDf

def visualize_detection(dfrow, category):
    mlist = dfrow[getKpKeys(category)]
    imageid = mlist[0]
    kplst = mlist[1:]

    # load cvmat
    xcvmat = cv2.imread(os.path.join("../../data/test", imageid))

    kpAnnLst = list()
    for _kpstr in kplst:
        _kpAnn = KpAnno.readFromStr(_kpstr)
        kpAnnLst.append(_kpAnn)

    # draw kp point on image
    mcvmat = visualize_predicted_keypoint(xcvmat, kpAnnLst, (255, 0, 0))
    cv2.imshow(category, mcvmat)
    cv2.waitKey()


def visualize_predicted_keypoint(cvmat, kpoints, color):
    for _kpAnn in kpoints:
        cv2.circle(cvmat, (_kpAnn.x, _kpAnn.y), radius=7, color=color, thickness=2)
    return cvmat

def submission_view(inputPath):
    csvfile = os.path.join(inputPath, "submission.csv")
    xdf = pd.read_csv(csvfile)
    for category in ['skirt', 'blouse', 'outwear', 'trousers', 'dress']:
        mdf = xdf[xdf['image_category'] == category]
        mdf = mdf.sample(frac=1)
        mdf = mdf[:10]
        for _index, _row in mdf.iterrows():
            visualize_detection(_row, category)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputPath", help='inputPath')
    parser.add_argument("--action", default='gen', help='action')
    args = parser.parse_args()

    if args.action == 'gen':
        submission(args.inputPath)
    elif args.action == 'view':
        submission_view(args.inputPath)
    else:
        assert (0), "only supported gen and view two actions"