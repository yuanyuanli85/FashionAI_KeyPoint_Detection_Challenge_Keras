
from refinenet import load_backbone_res101net, create_global_net_dilated, create_stack_refinenet
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras import backend as K
import keras

def Res101RefineNetMaskV3(n_classes, inputHeight, inputWidth, nStackNum):
    model = build_resnet101_stack_mask_v3(inputHeight, inputWidth, n_classes, nStackNum)
    return model

def euclidean_loss(x, y):
    return K.sqrt(K.sum(K.square(x - y)))

def apply_mask_to_output(output, mask):
    output_with_mask = keras.layers.multiply([output, mask])
    return output_with_mask

def build_resnet101_stack_mask_v3(inputHeight, inputWidth, n_classes, nStack):

    input_mask = Input(shape=(inputHeight//2, inputHeight//2, n_classes), name='mask')
    input_ohem_mask = Input(shape=(inputHeight//2, inputHeight//2, n_classes), name='ohem_mask')

    # backbone network
    input_image, lf2x,lf4x, lf8x, lf16x = load_backbone_res101net(inputHeight, inputWidth)

    # global net
    g8x, g4x, g2x = create_global_net_dilated((lf2x, lf4x, lf8x, lf16x), n_classes)

    s8x, s4x, s2x = g8x, g4x, g2x

    g2x_mask = apply_mask_to_output(g2x, input_mask)

    outputs =  [g2x_mask]
    for i in range(nStack):
        s8x, s4x, s2x =  create_stack_refinenet((s8x, s4x, s2x), n_classes, 'stack_'+str(i))
        if i == (nStack-1): # last stack with ohem_mask
            s2x_mask = apply_mask_to_output(s2x, input_ohem_mask)
        else:
            s2x_mask = apply_mask_to_output(s2x, input_mask)
        outputs.append(s2x_mask)

    model = Model(inputs=[input_image, input_mask, input_ohem_mask], outputs=outputs)

    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss=euclidean_loss, metrics=["accuracy"])
    return model