
from keras.models import *
from keras.layers import *
from keras import backend as K
from keras.utils import plot_model
from vgg_unet import VGGUnet, VGGUnetV2, VGGUnetV3, euclidean_loss
from fashion_net import FashionNet
from data_generator import DataGenerator
from dataset import getKpNum
import cv2
import utils
import datetime
import pandas as pd

def main_v1():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    xnet = VGGUnetV2(5 , 512, 512)
    plot_model(xnet, to_file='vggunet.png', show_shapes=True)

def main_v2():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    xmodel = VGGUnetV3(5 , 512, 512)
    print xmodel.output_shape, xmodel.input_shape

def main_train():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    category='skirt'
    xnet = FashionNet(512, 512, getKpNum(category)+1)
    xnet.build_model(modelName='v6', show=True)
    #xnet.train(category, epochs=50)


def main_retrain():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    category='skirt'
    xnet = FashionNet(512, 512, getKpNum(category)+1)
    xnet.load_model("../../trained_models/skirt/2018_03_28_11_27_14/skirt_weights_13.hdf5")
    xnet.train(category, batchSize=8)

def main_test():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    import tensorflow as tf
    sess = tf.Session()
    K.set_session(sess)

    category = 'skirt'
    valDt = DataGenerator(category, os.path.join("../../data/train/Annotations", "val_split.csv"))
    valGen = valDt.generator(mode='train', batchSize=1, inputSize=(512, 512),flatten=False)

    xnet = FashionNet(512, 512, getKpNum(category)+1)
    xnet.load_model(os.path.join("../../trained_models/2018_03_20_21_43", "skirt_adam_weights_45_39.33.hdf5"))

    with sess.as_default():
        for input, gtmap in valGen:
            cvmat = input[0, :, :, :] + 0.5
            cvmat = cv2.resize(cvmat, (256, 256))
            predmap = xnet.predict(input)
            for i in range(getKpNum(category)+1):
                mask = gtmap[0, :, :, i]
                premask =  predmap[0,:, :, i]

                #lossTensor = K.sum(K.square(premask - mask))/1024
                lossTensor = euclidean_loss(premask, mask)
                #with K.sess.as_default():
                #with K.get_session():
                print lossTensor, lossTensor.eval()
                K.print_tensor(lossTensor, 'loss b/w gt and predicated')

                # mask = mask*256
                # mask = mask.astype(np.uint8)
                cv2.imshow('image', cvmat)
                cv2.imshow('premask'+str(i), premask)
                cv2.imshow('mask' + str(i), mask)
                cv2.waitKey()
                # cv2.imwrite("mask"+str(i)+".png", mask)

def main_submission():
    import pickle
    from dataset import fill_dataframe
    # get full keys
    #"../../submission/skirt.pkl"
    xdf = pd.read_csv("../../data/train/Annotations/train.csv")
    print xdf.keys()

    with open("../../submission/skirt.pkl") as xfile:
        xpkl = pickle.load(xfile)

    outdf = pd.DataFrame(columns=xdf.keys(), index=np.arange(len(xpkl)))
    print outdf.keys()
    print len(outdf)

    outdf['image_id'] = xpkl.keys()

    outdf = outdf.fillna(value='-1_-1_-1')

    for _index, _row in outdf.iterrows():
        _imageId = _row['image_id']
        kplst = xpkl[_imageId]
        fill_dataframe(kplst, 'skirt', _row)

    print outdf[0:2]

    outdf.to_csv("../../submission/skirt.csv", index=False)


if __name__ == "__main__":
    main_retrain()