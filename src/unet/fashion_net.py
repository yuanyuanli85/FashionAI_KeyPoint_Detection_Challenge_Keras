
import sys
sys.path.insert(0, "../data_gen/")
sys.path.insert(0, "../eval/")


from vgg_unet import VGGUnetV2, VGGUnetV3, euclidean_loss
from data_generator import DataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from data_process import pad_image, normalize_image
import os
import cv2
import numpy as np
import datetime
from eval_callback import NormalizedErrorCallBack
from refinenet import VggRefineNet, VggRefineNetDilated, Res50RefineNetDilated, \
    Res101RefineNetDilated, Res101RefineNetStacked
from resnet101 import Scale
from lr_scheduler import LRScheduler
from stacknet import StackNet

class FashionNet(object):

    def __init__(self, inputHeight, inputWidth, nClasses):
        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.nClass = nClasses
        self.outputFlatten = False
        self.nStackNum = 1
        self.with_mask = False

    def build_model(self, modelName='v2', show=False):
        self.modelName = modelName
        self.outputFlatten = False
        if modelName == 'v2':
            self.model = VGGUnetV2(self.nClass, self.inputHeight, self.inputWidth, mode='train', frozenlayers=True)
        elif modelName == 'v3':
            self.model = VGGUnetV3(self.nClass, self.inputHeight, self.inputWidth, mode='train')
            self.outputFlatten = True
        elif modelName == 'v4':
            self.model = VggRefineNet(self.nClass, self.inputHeight, self.inputWidth)
        elif modelName == 'v5':
            self.model = VggRefineNetDilated(self.nClass, self.inputHeight, self.inputWidth)
        elif modelName == 'v6':
            self.model = Res50RefineNetDilated(self.nClass, self.inputHeight, self.inputWidth)
        elif modelName == 'v7':
            self.model = Res101RefineNetDilated(self.nClass, self.inputHeight, self.inputWidth)
            self.nStackNum = 1
        elif modelName == 'v8':
            self.model = Res101RefineNetStacked(self.nClass, self.inputHeight, self.inputWidth, nStackNum=2)
            # todo : fixme not hard code
            self.nStackNum = 2
        elif modelName == 'v9':
            self.model = StackNet(self.nClass, n_stacks=2)
            self.nStackNum = 1
            self.with_mask = True
        else:
            assert (0), "illegal network model"
        # show model summary and layer name
        if show:
            self.model.summary()
            for layer in self.model.layers:
                print layer.name, layer.trainable

    def train(self, category, batchSize=8, epochs=20, lrschedule=False):
        # fixme: disable flipflag for easy compare
        trainDt = DataGenerator(category, os.path.join("../../data/train/Annotations", "train_split.csv"))
        if self.with_mask:
            trainGen = trainDt.generator_with_mask(mode='train', batchSize=batchSize,
                                                   inputSize=(self.inputHeight, self.inputWidth),
                                                   nStackNum=self.nStackNum, flipFlag=False)
        else:
            trainGen = trainDt.generator(mode='train', batchSize=batchSize,
                                         inputSize=(self.inputHeight, self.inputWidth),
                                         flatten=self.outputFlatten, refineNetFlag=True, nStackNum=self.nStackNum)

        valDt = DataGenerator(category, os.path.join("../../data/train/Annotations", "val_split.csv"))
        if self.with_mask:
            valGen = valDt.generator_with_mask(mode='val', batchSize=batchSize,
                                               inputSize=(self.inputHeight, self.inputWidth),
                                               nStackNum=self.nStackNum, flipFlag=False)
        else:
            valGen = valDt.generator(mode='val', batchSize=batchSize, inputSize=(self.inputHeight, self.inputWidth),
                                     flatten=self.outputFlatten, refineNetFlag=True, nStackNum=self.nStackNum)

        normalizedErrorCallBack = NormalizedErrorCallBack("../../trained_models/", category, True)

        csvlogger = CSVLogger(os.path.join(normalizedErrorCallBack.get_folder_path(),
                                           "csv_train_" + self.modelName + "_" + str(
                                               datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        xcallbacks = [normalizedErrorCallBack, csvlogger]
        if lrschedule:
            lrScheduler = LRScheduler(0.5, 10)  # decay for 10 epochs
            xcallbacks.append(lrScheduler)

        self.model.fit_generator(generator=trainGen, steps_per_epoch=trainDt.get_dataset_size() // batchSize,
                                 validation_data=valGen, validation_steps=valDt.get_dataset_size() // batchSize,
                                 epochs=epochs, callbacks=xcallbacks)

    def load_model(self, netWeightFile):
        self.model = load_model(netWeightFile, custom_objects={'euclidean_loss': euclidean_loss, 'Scale': Scale})

    def resume_train(self, category, pretrainModel, modelName, initEpoch, batchSize=8, epochs=20):
        self.modelName = modelName
        self.load_model(pretrainModel)
        refineNetflag = True

        modelPath = os.path.dirname(pretrainModel)

        trainDt = DataGenerator(category, os.path.join("../../data/train/Annotations", "train_split.csv"))
        trainGen = trainDt.generator(mode='train', batchSize=batchSize, inputSize=(self.inputHeight, self.inputWidth),
                                     flatten=self.outputFlatten, refineNetFlag=refineNetflag)

        valDt = DataGenerator(category, os.path.join("../../data/train/Annotations", "val_split.csv"))
        valGen = valDt.generator(mode='val', batchSize=batchSize, inputSize=(self.inputHeight, self.inputWidth),
                                 flatten=self.outputFlatten, refineNetFlag=refineNetflag)


        normalizedErrorCallBack = NormalizedErrorCallBack("../../trained_models/", category, refineNetflag, resumeFolder=modelPath)

        csvlogger = CSVLogger(os.path.join(normalizedErrorCallBack.get_folder_path(),
                                           "csv_train_" + self.modelName + "_" + str(
                                               datetime.datetime.now().strftime('%H:%M')) + ".csv"))

        self.model.fit_generator(initial_epoch=initEpoch, generator=trainGen, steps_per_epoch=trainDt.get_dataset_size() // batchSize,
                                 validation_data=valGen, validation_steps=valDt.get_dataset_size() // batchSize,
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