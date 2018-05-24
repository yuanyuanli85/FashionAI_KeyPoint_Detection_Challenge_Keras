import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../unet/")

import argparse
import os
from fashion_net import FashionNet
from dataset import getKpNum
import tensorflow as tf
from keras import backend as k

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpuID", default=0, type=int, help='gpu id')
    parser.add_argument("--category", help="specify cloth category")
    parser.add_argument("--network", help="specify  network arch'")
    parser.add_argument("--batchSize", default=8, type=int, help='batch size for training')
    parser.add_argument("--epochs", default=20, type=int, help="number of traning epochs")
    parser.add_argument("--resume", default=False, type=bool,  help="resume training or not")
    parser.add_argument("--lrdecay", default=False, type=bool,  help="lr decay or not")
    parser.add_argument("--resumeModel", help="start point to retrain")
    parser.add_argument("--initEpoch", type=int, help="epoch to resume")


    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)


    # TensorFlow wizardry
    config = tf.ConfigProto()

    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True

    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 1.0

    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=config))

    if not args.resume :
        xnet = FashionNet(512, 512, getKpNum(args.category))
        xnet.build_model(modelName=args.network, show=True)
        xnet.train(args.category, epochs=args.epochs, batchSize=args.batchSize, lrschedule=args.lrdecay)
    else:
        xnet = FashionNet(512, 512, getKpNum(args.category))
        xnet.resume_train(args.category, args.resumeModel, args.network, args.initEpoch,
                          epochs=args.epochs, batchSize=args.batchSize)