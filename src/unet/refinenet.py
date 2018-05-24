from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.applications.resnet50 import ResNet50

IMAGE_ORDERING = 'channels_last'

def Res101RefineNetDilated(n_classes, inputHeight, inputWidth):
    model = build_network_resnet101(inputHeight, inputWidth, n_classes, dilated=True)
    return model

def Res101RefineNetStacked(n_classes, inputHeight, inputWidth, nStackNum):
    model = build_network_resnet101_stack(inputHeight, inputWidth, n_classes, nStackNum)
    return model

def euclidean_loss(x, y):
    return K.sqrt(K.sum(K.square(x - y)))


def create_global_net(lowlevelFeatures, n_classes):
    lf2x, lf4x, lf8x, lf16x = lowlevelFeatures

    o = lf16x

    o = (Conv2D(256, (3, 3), activation='relu', padding='same', name='up16x_conv', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), name='upsample_16x', activation='relu', padding='same',
                    data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, lf8x], axis=-1))
    o = (Conv2D(128, (3, 3), activation='relu', padding='same', name='up8x_conv', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    fup8x = o

    o = (Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name='upsample_8x', padding='same', activation='relu',
                         data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, lf4x], axis=-1))
    o = (Conv2D(64, (3, 3), activation='relu', padding='same', name='up4x_conv', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    fup4x = o

    o = (Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), name='upsample_4x', padding='same', activation='relu',
                         data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, lf2x], axis=-1))
    o = (Conv2D(64, (3, 3), activation='relu', padding='same', name='up2x_conv', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    fup2x = o

    out2x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='out2x', data_format=IMAGE_ORDERING)(fup2x)
    out4x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='out4x', data_format=IMAGE_ORDERING)(fup4x)
    out8x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='out8x', data_format=IMAGE_ORDERING)(fup8x)

    x4x = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(out8x)
    eadd4x = Add(name='global4x')([x4x, out4x])

    x2x = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(eadd4x)
    eadd2x = Add(name='global2x')([x2x, out2x])

    return (fup8x, eadd4x, eadd2x)

def create_refine_net(inputFeatures, n_classes):
    f8x, f4x, f2x = inputFeatures

    # 2 Conv2DTranspose f8x -> fup8x
    fup8x = (Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name='refine8x_deconv_1', padding='same', activation='relu',
                         data_format=IMAGE_ORDERING))(f8x)
    fup8x = (BatchNormalization())(fup8x)

    fup8x = (Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name='refine8x_deconv_2', padding='same', activation='relu',
                         data_format=IMAGE_ORDERING))(fup8x)
    fup8x = (BatchNormalization())(fup8x)

    # 1 Conv2DTranspose f4x -> fup4x
    fup4x = (Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name='refine4x_deconv', padding='same', activation='relu',
                    data_format=IMAGE_ORDERING))(f4x)

    fup4x = (BatchNormalization())(fup4x)

    # 1 conv f2x -> fup2x
    fup2x =  (Conv2D(128, (3, 3), activation='relu', padding='same', name='refine2x_conv', data_format=IMAGE_ORDERING))(f2x)
    fup2x =  (BatchNormalization())(fup2x)

    # concat f2x, fup8x, fup4x
    fconcat = (concatenate([fup8x, fup4x, fup2x], axis=-1, name='refine_concat'))

    # 1x1 to map to required feature map
    out2x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='refine2x', data_format=IMAGE_ORDERING)(fconcat)

    return out2x


def create_refine_net_bottleneck(inputFeatures, n_classes):
    f8x, f4x, f2x = inputFeatures

    # 2 Conv2DTranspose f8x -> fup8x
    fup8x = (Conv2D(256, kernel_size=(1, 1),  name='refine8x_1', padding='same', activation='relu', data_format=IMAGE_ORDERING))(f8x)
    fup8x = (BatchNormalization())(fup8x)

    fup8x = (Conv2D(128, kernel_size=(1, 1),  name='refine8x_2', padding='same', activation='relu', data_format=IMAGE_ORDERING))(fup8x)
    fup8x = (BatchNormalization())(fup8x)

    fup8x = UpSampling2D((4, 4), data_format=IMAGE_ORDERING)(fup8x)


    # 1 Conv2DTranspose f4x -> fup4x
    fup4x = (Conv2D(128, kernel_size=(1, 1), name='refine4x', padding='same', activation='relu', data_format=IMAGE_ORDERING))(f4x)
    fup4x = (BatchNormalization())(fup4x)
    fup4x = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(fup4x)


    # 1 conv f2x -> fup2x
    fup2x =  (Conv2D(128, (1, 1), activation='relu', padding='same', name='refine2x_conv', data_format=IMAGE_ORDERING))(f2x)
    fup2x =  (BatchNormalization())(fup2x)

    # concat f2x, fup8x, fup4x
    fconcat = (concatenate([fup8x, fup4x, fup2x], axis=-1, name='refine_concat'))

    # 1x1 to map to required feature map
    out2x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='refine2x', data_format=IMAGE_ORDERING)(fconcat)

    return out2x


def create_stack_refinenet(inputFeatures, n_classes, layerName):
    f8x, f4x, f2x = inputFeatures

    # 2 Conv2DTranspose f8x -> fup8x
    fup8x = (Conv2D(256, kernel_size=(1, 1), name=layerName+'_refine8x_1', padding='same', activation='relu'))(f8x)
    fup8x = (BatchNormalization())(fup8x)

    fup8x = (Conv2D(128, kernel_size=(1, 1), name=layerName+'refine8x_2', padding='same', activation='relu'))(fup8x)
    fup8x = (BatchNormalization())(fup8x)

    out8x = fup8x
    fup8x = UpSampling2D((4, 4), data_format=IMAGE_ORDERING)(fup8x)

    # 1 Conv2DTranspose f4x -> fup4x
    fup4x = (Conv2D(128, kernel_size=(1, 1), name=layerName+'refine4x', padding='same', activation='relu'))(f4x)
    fup4x = (BatchNormalization())(fup4x)
    out4x = fup4x
    fup4x = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(fup4x)

    # 1 conv f2x -> fup2x
    fup2x = (Conv2D(128, (1, 1), activation='relu', padding='same', name=layerName+'refine2x_conv'))(f2x)
    fup2x = (BatchNormalization())(fup2x)

    # concat f2x, fup8x, fup4x
    fconcat = (concatenate([fup8x, fup4x, fup2x], axis=-1, name=layerName+'refine_concat'))

    # 1x1 to map to required feature map
    out2x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name=layerName+'refine2x')(fconcat)

    return out8x, out4x, out2x


def create_global_net_dilated(lowlevelFeatures, n_classes):
    lf2x, lf4x, lf8x, lf16x = lowlevelFeatures

    o = lf16x

    o = (Conv2D(256, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='up16x_conv', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), name='upsample_16x', activation='relu', padding='same',
                    data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, lf8x], axis=-1))
    o = (Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='up8x_conv', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    fup8x = o

    o = (Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name='upsample_8x', padding='same', activation='relu',
                         data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, lf4x], axis=-1))
    o = (Conv2D(64, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='up4x_conv', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    fup4x = o

    o = (Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), name='upsample_4x', padding='same', activation='relu',
                         data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, lf2x], axis=-1))
    o = (Conv2D(64, (3, 3), dilation_rate=(2, 2), activation='relu', padding='same', name='up2x_conv', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
    fup2x = o

    out2x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='out2x', data_format=IMAGE_ORDERING)(fup2x)
    out4x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='out4x', data_format=IMAGE_ORDERING)(fup4x)
    out8x = Conv2D(n_classes, (1, 1), activation='linear', padding='same', name='out8x', data_format=IMAGE_ORDERING)(fup8x)

    x4x = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(out8x)
    eadd4x = Add(name='global4x')([x4x, out4x])

    x2x = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(eadd4x)
    eadd2x = Add(name='global2x')([x2x, out2x])

    return (fup8x, eadd4x, eadd2x)


def build_network_resnet101(inputHeight, inputWidth, n_classes, frozenlayers=True, dilated=False):
    input, lf2x, lf4x, lf8x, lf16x = load_backbone_res101net(inputHeight, inputWidth)

    # global net 8x, 4x, and 2x
    if dilated:
        g8x, g4x, g2x = create_global_net_dilated((lf2x, lf4x, lf8x, lf16x), n_classes)
    else:
        g8x, g4x, g2x = create_global_net((lf2x, lf4x, lf8x, lf16x), n_classes)

    # refine net, only 2x as output
    refine2x = create_refine_net_bottleneck((g8x, g4x, g2x), n_classes)

    model = Model(inputs=input, outputs=[g2x, refine2x])

    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss=euclidean_loss, metrics=["accuracy"])

    return model


def build_network_resnet101_stack(inputHeight, inputWidth, n_classes, nStack):
    # backbone network
    input, lf2x,lf4x, lf8x, lf16x = load_backbone_res101net(inputHeight, inputWidth)

    # global net
    g8x, g4x, g2x = create_global_net_dilated((lf2x, lf4x, lf8x, lf16x), n_classes)

    s8x, s4x, s2x = g8x, g4x, g2x

    outputs =  [g2x]
    for i in range(nStack):
        s8x, s4x, s2x =  create_stack_refinenet((s8x, s4x, s2x), n_classes, 'stack_'+str(i))
        outputs.append(s2x)

    model = Model(inputs=input, outputs=outputs)

    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss=euclidean_loss, metrics=["accuracy"])
    return model


def load_backbone_res101net(inputHeight, inputWidth):
    from resnet101 import ResNet101
    xresnet = ResNet101(weights='imagenet', include_top=False, input_shape=(inputHeight, inputWidth, 3))

    xresnet.load_weights("../../data/resnet101_weights_tf.h5", by_name=True)

    lf16x = xresnet.get_layer('res4b22_relu').output
    lf8x = xresnet.get_layer('res3b2_relu').output
    lf4x = xresnet.get_layer('res2c_relu').output
    lf2x = xresnet.get_layer('conv1_relu').output

    # add one padding for lf4x whose shape is 127x127
    lf4xp = ZeroPadding2D(padding=((0, 1), (0, 1)))(lf4x)

    return (xresnet.input, lf2x, lf4xp, lf8x, lf16x)