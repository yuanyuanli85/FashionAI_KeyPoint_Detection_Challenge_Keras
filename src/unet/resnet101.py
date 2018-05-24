# -*- coding: utf-8 -*-
"""ResNet-101 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Slightly modified Felix Yu's (https://github.com/flyyufelix) implementation of
ResNet-101 to have consistent API as those pre-trained models within
`keras.applications`. The original implementation is found here
https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294#file-resnet-101_keras-py

Implementation is based on Keras 2.0
"""
from keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D,
    Flatten, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D, add)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file

import warnings
import sys
sys.setrecursionlimit(3000)


WEIGHTS_PATH_TH = 'https://dl.dropboxusercontent.com/s/rrp56zm347fbrdn/resnet101_weights_th.h5?dl=0'
WEIGHTS_PATH_TF = 'https://dl.dropboxusercontent.com/s/a21lyqwgf88nz9b/resnet101_weights_tf.h5?dl=0'
MD5_HASH_TH = '3d2e9a49d05192ce6e22200324b7defe'
MD5_HASH_TF = '867a922efc475e9966d0f3f7b884dc15'


class Scale(Layer):
    '''Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights`
            argument.
    '''
    def __init__(self,
                 weights=None,
                 axis=-1,
                 momentum=0.9,
                 beta_init='zero',
                 gamma_init='one',
                 **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(
            self.gamma_init(shape),
            name='{}_gamma'.format(self.name))
        self.beta = K.variable(
            self.beta_init(shape),
            name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(
            self.gamma,
            broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main
            path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
               use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,
                           name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,
                           name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
               use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,
                           name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main
            path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with
    strides=(2,2). And the shortcut should have strides=(2,2) as well
    '''
    eps = 1.1e-5
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,
                           name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,
                           name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1),
               name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,
                           name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis,
                                  name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def ResNet101(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000):
    """Instantiates the ResNet-101 architecture.

    Optionally loads weights pre-trained on ImageNet. Note that when using
    TensorFlow, for best performance you should set
    image_data_format='channels_last'` in your Keras config at
    ~/.keras/keras.json.

    The model and the weights are compatible with both TensorFlow and Theano.
    The data format convention used by the model is the one specified in your
    Keras config file.

    Parameters
    ----------
        include_top: whether to include the fully-connected layer at the top of
            the network.
        weights: one of `None` (random initialization) or 'imagenet'
            (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified if
            `include_top` is False (otherwise the input shape has to be
            `(224, 224, 3)` (with `channels_last` data format) or
            `(3, 224, 224)` (with `channels_first` data format). It should have
            exactly 3 inputs channels, and width and height should be no
            smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction when
            `include_top` is `False`.
            - `None` means that the output of the model will be the 4D tensor
                output of the last convolutional layer.
            - `avg` means that global average pooling will be applied to the
                output of the last convolutional layer, and thus the output of
                the model will be a 2D tensor.
            - `max` means that global max pooling will be applied.
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True, and if no `weights` argument is
            specified.

    Returns
    -------
        A Keras model instance.

    Raises
    ------
        ValueError: in case of invalid argument for `weights`, or invalid input
        shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(
                tensor=input_tensor, shape=input_shape, name='data')
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    eps = 1.1e-5

    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 3):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 23):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='mmfc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet101')

    '''
    # load weights
    if weights == 'imagenet':
        filename = 'resnet101_weights_{}.h5'.format(K.image_dim_ordering())
        if K.backend() == 'theano':
            path = WEIGHTS_PATH_TH
            md5_hash = MD5_HASH_TH
        else:
            path = WEIGHTS_PATH_TF
            md5_hash = MD5_HASH_TF
        weights_path = get_file(
            fname=filename,
            origin=path,
            cache_subdir='models',
            md5_hash=md5_hash,
            hash_algorithm='md5')
        model.load_weights(weights_path, by_name=True)

        if K.image_data_format() == 'channels_first' and K.backend() == 'tensorflow':
            warnings.warn('You are using the TensorFlow backend, yet you '
                          'are using the Theano '
                          'image data format convention '
                          '(`image_data_format="channels_first"`). '
                          'For best performance, set '
                          '`image_data_format="channels_last"` in '
                          'your Keras config '
                          'at ~/.keras/keras.json.')
    '''
    return model
