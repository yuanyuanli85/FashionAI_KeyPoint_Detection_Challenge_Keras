from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from refinenet import euclidean_loss
from resnet101 import Scale

BASE_NETWORK_MODEL_FILE_0413= "../../trained_models/all/2018_04_09_10_46_41_x2/all_weights_11.hdf5"
'''
overall: 0.08504728270395223
blouse: 0.11236836376150304
skirt: 0.06213196024044802
outwear: 0.07234653211887554
dress: 0.0717178216984672
trousers: 0.08415049454342481
'''

BASE_NETWORK_MODEL_FILE= "../../trained_models/all/2018_04_13_11_56_21/all_weights_15.hdf5"

'''
overall: 0.0684568940979324
blouse: 0.07220305409835072
skirt: 0.05912974928341943
outwear: 0.06531115712293527
dress: 0.07029947782195266
trousers: 0.06956724604759555
'''

def StackNet(n_classes, n_stacks):
    model = exp_create_cascade_network(n_classes, n_stacks)
    return model

def exp_create_cascade_network(n_classes, n_stacks):
    input, lf8x, lf4x, lf2x = load_all_in_one_network(BASE_NETWORK_MODEL_FILE, frozen=False)

    preStackOut = (lf8x, lf4x, lf2x)
    outputs = list()
    for i in range(n_stacks):
        _f8x, _f4x, _f2x = create_stack_refinenet((preStackOut), n_classes, 's'+str(i))
        #_f8x, _f4x, _f2x = exp_create_stack_refinenet((preStackOut), n_classes, 's' + str(i))
        outputs.append(_f2x)
        preStackOut = (_f8x, _f4x, _f2x)

    model = Model(inputs=input, outputs=outputs)

    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss=euclidean_loss, metrics=["accuracy"])
    return model


def load_all_in_one_network(modelfile, frozen=True):

    model = load_model(modelfile, custom_objects={'euclidean_loss': euclidean_loss, 'Scale': Scale})

    if frozen:
        for layer in model.layers:
            layer.trainable = False

    lf8x = model.get_layer('stack_1refine8x_2').output
    lf4x = model.get_layer('stack_1refine4x').output
    lf2x = model.get_layer('stack_1refine2x_conv').output

    return (model.input, lf8x, lf4x, lf2x)


def create_stack_refinenet(inputFeatures, n_classes, layerName):
    f8x, f4x, f2x = inputFeatures

    # 2 Conv2DTranspose f8x -> fup8x
    fup8x = (Conv2D(256, kernel_size=(1, 1), name=layerName+'_refine8x_1', padding='same', activation='relu'))(f8x)
    fup8x = (BatchNormalization(name=layerName+'_refine8x_bn_1'))(fup8x)

    fup8x = (Conv2D(128, kernel_size=(1, 1), name=layerName+'_refine8x_2', padding='same', activation='relu'))(fup8x)
    fup8x = (BatchNormalization(name=layerName+'_refine8x_bn_2'))(fup8x)

    out8x = fup8x
    fup8x = UpSampling2D((4, 4), name=layerName+'_refine8x_up') (fup8x)

    # 1 Conv2DTranspose f4x -> fup4x
    fup4x = (Conv2D(128, kernel_size=(1, 1), name=layerName+'_refine4x', padding='same', activation='relu'))(f4x)
    fup4x = (BatchNormalization(name=layerName+'_refine4x_bn'))(fup4x)
    out4x = fup4x
    fup4x = UpSampling2D((2, 2), name=layerName+'_refine4x_up')(fup4x)

    # 1 conv f2x -> fup2x
    fup2x = (Conv2D(128, (1, 1), activation='relu', padding='same', name=layerName+'_refine2x_conv'))(f2x)
    fup2x = (BatchNormalization(name=layerName+'_refine2x_bn'))(fup2x)

    # concat f2x, fup8x, fup4x
    fconcat = (concatenate([fup8x, fup4x, fup2x], axis=-1, name=layerName+'_refine2x_concat'))

    # 1x1 to map to required feature map
    out2x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name=layerName+'_refine2x')(fconcat)

    return out8x, out4x, out2x

def exp_create_stack_refinenet(inputFeatures, n_classes, layerName):
    f8x, f4x, f2x = inputFeatures

    # 2 Conv2DTranspose f8x -> fup8x
    fup8x = (Conv2D(256, kernel_size=(3, 3), name=layerName+'_refine8x_1', padding='same', activation='relu'))(f8x)
    fup8x = (BatchNormalization(name=layerName+'_refine8x_bn_1'))(fup8x)

    fup8x = (Conv2D(128, kernel_size=(3, 3), name=layerName+'_refine8x_2', padding='same', activation='relu'))(fup8x)
    fup8x = (BatchNormalization(name=layerName+'_refine8x_bn_2'))(fup8x)

    fup8x = (Conv2D(128, kernel_size=(3, 3), name=layerName+'_refine8x_3', padding='same', activation='relu'))(fup8x)
    fup8x = (BatchNormalization(name=layerName+'_refine8x_bn_3'))(fup8x)

    out8x =  fup8x
    fup8x =  Conv2DTranspose(128, kernel_size=(3, 3), strides=(4, 4), name=layerName+'_deconv_8x',
                             activation='relu', padding='same')(fup8x)


    fup4x = (Conv2D(128, kernel_size=(3, 3), name=layerName+'_refine4x_1', padding='same', activation='relu'))(f4x)
    fup4x = (BatchNormalization(name=layerName+'_refine4x_bn_1'))(fup4x)

    fup4x = (Conv2D(128, kernel_size=(3, 3), name=layerName+'_refine4x_2', padding='same', activation='relu'))(f4x)
    fup4x = (BatchNormalization(name=layerName+'_refine4x_bn_2'))(fup4x)

    out4x = fup4x
    fup4x = Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name=layerName + '_deconv_4x',
                            activation='relu', padding='same')(fup4x)

    # 1 conv f2x -> fup2x
    fup2x = (Conv2D(128, (3, 3), activation='relu', padding='same', name=layerName+'_refine2x_conv'))(f2x)
    fup2x = (BatchNormalization(name=layerName+'_refine2x_bn'))(fup2x)

    # concat f2x, fup8x, fup4x
    fconcat = (concatenate([fup8x, fup4x, fup2x], axis=-1, name=layerName+'_refine2x_concat'))

    # 1x1 to map to required feature map
    out2x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name=layerName+'_refine2x')(fconcat)

    return out8x, out4x, out2x

if __name__ == "__main__":
    from keras.utils import plot_model
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = exp_create_cascade_network(5, 2)
    model.summary()
    plot_model(model, 'stack.png', show_shapes=True, show_layer_names=True)
