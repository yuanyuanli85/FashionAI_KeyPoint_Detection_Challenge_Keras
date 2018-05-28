import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../eval/")
sys.path.insert(0, "../unet/")

import argparse
import os
import pandas as pd
import cv2
from evaluation import Evaluation
from dataset import getKpKeys, get_kp_index_from_allkeys

def visualize_keypoint(imageName, category, dtkp):
    cvmat = cv2.imread(imageName)
    for key in getKpKeys(category)[1:]:
        index = get_kp_index_from_allkeys(key)
        _kp = dtkp[index]
        cv2.circle(cvmat, center=(_kp.x, _kp.y), radius=7, color=(1.0, 0.0, 0.0), thickness=2)
    cv2.imshow('demo', cvmat)
    cv2.waitKey()

def demo(modelfile):

    # load network
    xEval = Evaluation('all', modelfile)

    # load images and run prediction
    testfile = os.path.join("../../data/test/", 'test.csv')
    xdf = pd.read_csv(testfile)
    xdf = xdf.sample(frac=1.0)

    for _index, _row in xdf.iterrows():
        _image_id = _row['image_id']
        _category = _row['image_category']
        imageName = os.path.join("../../data/test", _image_id)
        print _image_id, _category
        dtkp = xEval.predict_kp_with_rotate(imageName, _category)
        visualize_keypoint(imageName, _category, dtkp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--modelfile", help="file of model")

    args = parser.parse_args()

    print args

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    demo(args.modelfile)