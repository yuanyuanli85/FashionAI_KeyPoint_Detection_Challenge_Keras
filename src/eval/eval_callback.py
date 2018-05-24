
import keras
import os
import datetime
from evaluation import Evaluation
from time import time
class NormalizedErrorCallBack(keras.callbacks.Callback):

    def __init__(self, foldpath, category, multiOut=False, resumeFolder=None):
        self.parentFoldPath = foldpath
        self.category = category

        if resumeFolder is None:
            self.foldPath = os.path.join(self.parentFoldPath, self.category, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            if not os.path.exists(self.foldPath):
                os.mkdir(self.foldPath)
        else:
            self.foldPath = resumeFolder

        self.valLog = os.path.join(self.foldPath, 'val.log')
        self.multiOut = multiOut

    def get_folder_path(self):
        return self.foldPath

    def on_epoch_end(self, epoch, logs=None):
        modelName = os.path.join(self.foldPath, self.category+"_weights_"+str(epoch)+".hdf5")
        keras.models.save_model(self.model, modelName)
        print "Saving model to ", modelName

        print "Runing evaluation ........."

        xEval = Evaluation(self.category, None)
        xEval.init_from_model(self.model)

        start = time()
        neScore, categoryDict = xEval.eval(self.multiOut, details=True)
        end = time()
        print "Evaluation Done", str(neScore), " cost ", end - start, " seconds!"

        for key in categoryDict.keys():
            scores = categoryDict[key]
            print key, ' score ', sum(scores)/len(scores)

        with open(self.valLog , 'a+') as xfile:
            xfile.write(modelName + ", Socre "+ str(neScore)+"\n")
            for key in categoryDict.keys():
                scores = categoryDict[key]
                xfile.write(key + ": " + str(sum(scores)/len(scores)) + "\n")

        xfile.close()