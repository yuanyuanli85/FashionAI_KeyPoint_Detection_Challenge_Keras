
import sys
sys.path.insert(0, "../unet/")

from keras.models import *
from keras.layers import *
from utils import np_euclidean_l2
from dataset import getKpNum

def generate_topk_mask_ohem(input_data, gthmap, keras_model, graph, topK, image_category, dynamicFlag=False):
    '''
    :param input_data: input
    :param gthmap:  ground truth
    :param keras_model: keras model
    :param graph:  tf grpah to WA thread issue
    :param topK: number of kp selected
    :return:
    '''

    # do inference, and calculate loss of each channel
    mimg, mmask = input_data
    ximg  = mimg[np.newaxis,:,:,:]
    xmask = mmask[np.newaxis,:,:,:]

    if len(keras_model.input_layers) == 3:
        # use original mask as ohem_mask
        inputs = [ximg, xmask, xmask]
    else:
        inputs = [ximg, xmask]

    with graph.as_default():
        keras_output = keras_model.predict(inputs)

    # heatmap of last stage
    outhmap = keras_output[-1]

    channel_num = gthmap.shape[-1]

    # calculate loss
    mloss = list()
    for i in range(channel_num):
        _dtmap = outhmap[0, :, :, i]
        _gtmap = gthmap[:, :, i]
        loss   = np_euclidean_l2(_dtmap, _gtmap)
        mloss.append(loss)

    # refill input_mask, set topk as 1.0 and fill 0.0 for rest
    # fixme: topk may different b/w category
    if dynamicFlag:
        topK = getKpNum(image_category)//2

    ohem_mask   = adjsut_mask(mloss, mmask, topK)

    ohem_gthmap = ohem_mask * gthmap

    return ohem_mask, ohem_gthmap

def adjsut_mask(loss, input_mask,  topk):
    # pick topk loss from losses
    # fill topk with 1.0 and fill the rest as 0.0
    assert (len(loss) == input_mask.shape[-1]), \
        "shape should be same" + str(len(loss)) + " vs " + str(input_mask.shape)

    outmask = np.zeros(input_mask.shape, dtype=np.float)

    topk_index = sorted(range(len(loss)), key=lambda i:loss[i])[-topk:]

    for i in range(len(loss)):
        if i in topk_index:
            outmask[:,:,i] = 1.0

    return outmask
