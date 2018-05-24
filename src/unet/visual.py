import argparse
import os
import cv2
from fashion_net import FashionNet
from dataset import getKpNum
from data_generator import DataGenerator
import keras.backend as K
from vgg_unet import euclidean_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--category", help="specify cloth category")
    parser.add_argument("--modelFile", help="model file path")
    parser.add_argument("--onlyBackground", default=True, type=bool, help='show only background')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

    xnet = FashionNet(512, 512, getKpNum(args.category) + 1)
    xnet.load_model(args.modelFile)

    category = args.category
    valDt = DataGenerator(category, os.path.join("../../data/train/Annotations", "val_split.csv"))
    valGen = valDt.generator(mode='train', batchSize=1, inputSize=(512, 512), flatten=False, rotateFlag=False)

    for input, gtmap in valGen:
        cvmat = input[0, :, :, :] + 0.5
        cvmat = cv2.resize(cvmat, (256, 256))
        predmap = xnet.predict(input)

        cv2.imshow('image', cvmat)
        if args.onlyBackground == False:
            for i in range(getKpNum(category)):
                mask = gtmap[0, :, :, i]
                premask = predmap[0, :, :, i]
                cv2.imshow('partmask' + str(i), premask)
                cv2.imshow('gtmask' + str(i), mask)

        backgroundMask = predmap[0, :, :, -1]
        cv2.imshow('background', backgroundMask)
        cv2.waitKey()

