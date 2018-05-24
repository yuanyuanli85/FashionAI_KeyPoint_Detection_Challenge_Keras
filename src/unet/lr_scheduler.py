
import keras
import keras.backend as K

class LRScheduler(keras.callbacks.Callback):

    def __init__(self, decay, epochStep):
        super(LRScheduler, self).__init__()
        self.decay = decay
        self.epochStep = epochStep

    def on_epoch_begin(self, epoch, logs=None):
        curLR = K.get_value(self.model.optimizer.lr)
        if epoch > 1 and epoch%self.epochStep == 0:
            K.set_value(self.model.optimizer.lr, curLR*self.decay)
        print epoch, 'cur lr', curLR, 'new lr', K.get_value(self.model.optimizer.lr)