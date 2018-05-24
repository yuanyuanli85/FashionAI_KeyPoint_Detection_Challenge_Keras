
import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../eval/")

from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from data_process import pad_image, normalize_image
import os
import cv2
import numpy as np
import datetime
from eval_callback import NormalizedErrorCallBack
from refinenet_mask_v3 import Res101RefineNetMaskV3, euclidean_loss
from resnet101 import Scale
import tensorflow as tf

class FashionNet(object):

    def __init__(self, inputHeight, inputWidth, nClasses):
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.nClass = nClasses

    def build_model(self, modelName='v2', show=False):
        self.modelName = modelName
        self.model = Res101RefineNetMaskV3(self.nClass, self.inputHeight, self.inputWidth, nStackNum=2)
        self.nStackNum = 2

        # show model summary and layer name
        if show:
            self.model.summary()
            for layer in self.model.layers:
                print layer.name, layer.trainable

    def train(self, category, batchSize=8, epochs=20, lrschedule=False):
        trainDt = DataGenerator(category, os.path.join("../../data/train/Annotations", "train_split.csv"))
        trainGen = trainDt.generator_with_mask_ohem( graph=tf.get_default_graph(), kerasModel=self.model,
                                    batchSize= batchSize, inputSize=(self.inputHeight, self.inputWidth),
                                    nStackNum=self.nStackNum, flipFlag=False, cropFlag=False)

        normalizedErrorCallBack = NormalizedErrorCallBack("../../trained_models/", category, True)

        csvlogger = CSVLogger( os.path.join(normalizedErrorCallBack.get_folder_path(),
                               "csv_train_"+self.modelName+"_"+str(datetime.datetime.now().strftime('%H:%M'))+".csv"))

        xcallbacks = [normalizedErrorCallBack, csvlogger]

        self.model.fit_generator(generator=trainGen, steps_per_epoch=trainDt.get_dataset_size()//batchSize,
                                 epochs=epochs,  callbacks=xcallbacks)

    def load_model(self, netWeightFile):
        self.model = load_model(netWeightFile, custom_objects={'euclidean_loss': euclidean_loss, 'Scale': Scale})

    def resume_train(self, category, pretrainModel, modelName, initEpoch, batchSize=8, epochs=20):
        self.modelName = modelName
        self.load_model(pretrainModel)
        refineNetflag = True
        self.nStackNum = 2

        modelPath = os.path.dirname(pretrainModel)

        trainDt = DataGenerator(category, os.path.join("../../data/train/Annotations", "train_split.csv"))
        trainGen = trainDt.generator_with_mask_ohem(graph=tf.get_default_graph(), kerasModel=self.model,
                                                    batchSize=batchSize, inputSize=(self.inputHeight, self.inputWidth),
                                                    nStackNum=self.nStackNum, flipFlag=False, cropFlag=False)


        normalizedErrorCallBack = NormalizedErrorCallBack("../../trained_models/", category, refineNetflag, resumeFolder=modelPath)

        csvlogger = CSVLogger(os.path.join(normalizedErrorCallBack.get_folder_path(),
                                           "csv_train_" + self.modelName + "_" + str(
                                               datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        self.model.fit_generator(initial_epoch=initEpoch, generator=trainGen, steps_per_epoch=trainDt.get_dataset_size() // batchSize,
                                 epochs=epochs, callbacks=[normalizedErrorCallBack, csvlogger])


    def predict_image(self, imgfile):
        # load image and preprocess
        img = cv2.imread(imgfile)
        img, _ = pad_image(img, list(), 512, 512)
        img = normalize_image(img)
        input = img[np.newaxis,:,:,:]
        # inference
        heatmap = self.model.predict(input)
        return heatmap


    def predict(self, input):
        # inference
        heatmap = self.model.predict(input)
        return heatmap