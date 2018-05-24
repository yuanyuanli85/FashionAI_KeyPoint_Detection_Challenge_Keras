
import sys
sys.path.insert(0, "../unet/")

from keras.models import *
from keras.layers import *
from utils import np_euclidean_l2

def ohem_topk(input_data, gthmap, keras_model, graph):

    # do inference, and calculate loss of each channel
    ximg, xmask = input_data
    ximg  = ximg[np.newaxis,:,:,:]
    xmask = xmask[np.newaxis,:,:,:]
    with graph.as_default():
        keras_output = keras_model.predict([ximg, xmask])

    outhmap = keras_output[-1]

    mloss = list()
    for i in range(gthmap.shape[-1]):
        _dtmap = outhmap[0, :, :, i]
        _gtmap = gthmap[:, :, i]
        loss   = np_euclidean_l2(_dtmap, _gtmap)
        mloss.append(loss)

    # refill input_mask, set topk as 1.0 and fill 0.0 for rest
    outmask = adjsut_mask(mloss, xmask, len(mloss)//2)

    gthmap = outmask * gthmap

    return outmask, gthmap


def adjsut_mask(loss, input_mask,  topk):
    # pick topk loss from losses
    # fill topk with 1.0 and fill the rest as 0.0
    assert (len(loss) == input_mask.shape[-1]), \
        "shape should be same" + str(len(loss)) + " vs " + str(input_mask.shape)

    topk_index = sorted(range(len(loss)), key=lambda i:loss[i])[-topk:]

    for i in range(len(loss)):
        if i in topk_index:
            input_mask[:,:,i] = 1.0
        else:
            input_mask[:,:,i] = 0.0

    return input_mask
