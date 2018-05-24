from keras.models import *
from keras.layers import *
from keras.optimizers import Adam, SGD
from keras import backend as K

import os
file_path = os.path.dirname( os.path.abspath(__file__) )


VGG_Weights_path = "../../data/vgg16_weights_tf_dim_ordering_tf_kernels.h5"

IMAGE_ORDERING = 'channels_last'


def euclidean_loss(x, y):
	return K.sqrt(K.sum(K.square(x - y)))


def VGGUnet( n_classes , input_height=512, input_width=512 , mode = 'train'):

	assert input_height%32 == 0
	assert input_width%32 == 0
	assert K.image_data_format()=='channels_last'

	img_input    = Input(shape=(input_height, input_width, 3))

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
	f5 = x

	## Modify the name of layers to avoid mismatched shape of dense layer
	x = Flatten(name='mflatten')(x)
	x = Dense(4096, activation='relu', name='mfc1')(x)
	x = Dense(4096, activation='relu', name='mfc2')(x)
	x = Dense( 1000 , activation='softmax', name='mpredictions')(x)

	## Set by_name=True to avoid loading non-existing layers
	vgg  = Model(img_input , x  )
	vgg.load_weights(VGG_Weights_path, by_name=True)

	levels = [f1 , f2 , f3 , f4 , f5 ]

	o = f4

	o = ( Conv2D(256, (3, 3), activation='relu', padding='same', name='up16x_conv', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	o = (Conv2DTranspose( 256, kernel_size=(3,3), strides=(2,2), name='upsample_16x', activation='relu',  padding='same', data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([ o ,f3],axis=-1 )  )
	o = ( Conv2D( 128, (3, 3), activation='relu', padding='same', name='up8x_conv', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	fup8x = o

	o = (Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name='upsample_8x', padding='same', activation='relu', data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([o,f2],axis=-1 ) )
	o = ( Conv2D( 64 , (3, 3), activation='relu', padding='same', name='up4x_conv', data_format=IMAGE_ORDERING ) )(o)
	o = ( BatchNormalization())(o)
	fup4x = o

	o = (Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2),  name='upsample_4x',padding='same', activation='relu', data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([o,f1],axis=-1 ) )
	o = ( Conv2D( 64 , (3, 3), activation='relu', padding='same', name='up2x_conv', data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)
	fup2x = o

	out2x = Conv2D(n_classes , (1, 1) , padding='same', name='out2x', data_format=IMAGE_ORDERING )( fup2x )
	out4x = Conv2D(n_classes, (1, 1), padding='same', name='out4x', data_format=IMAGE_ORDERING)( fup4x)
	out8x = Conv2D(n_classes, (1, 1), padding='same', name='out8x', data_format=IMAGE_ORDERING)( fup8x)

	model = Model(inputs=img_input, outputs=[out8x, out4x, out2x])
	model.compile(optimizer='Adam', loss=[euclidean_loss, euclidean_loss, euclidean_loss], metrics=["accuracy"])

	return model



def VGGUnetV2( n_classes , input_height=512, input_width=512 , mode = 'train', frozenlayers=True):

	assert input_height%32 == 0
	assert input_width%32 == 0
	assert K.image_data_format()=='channels_last'

	img_input    = Input(shape=(input_height, input_width, 3))

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
	f5 = x

	## Modify the name of layers to avoid mismatched shape of dense layer
	x = Flatten(name='mflatten')(x)
	x = Dense(4096, activation='relu', name='mfc1')(x)
	x = Dense(4096, activation='relu', name='mfc2')(x)
	x = Dense( 1000 , activation='softmax', name='mpredictions')(x)

	## Set by_name=True to avoid loading non-existing layers
	vgg  = Model(img_input , x  )
	vgg.load_weights(VGG_Weights_path, by_name=True)

	levels = [f1 , f2 , f3 , f4 , f5 ]

	o = f4

	o = ( Conv2D(256, (3, 3), activation='relu', padding='same', name='up16x_conv', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)

	o = (Conv2DTranspose( 256, kernel_size=(3,3), strides=(2,2), name='upsample_16x', activation='relu',  padding='same', data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([ o ,f3],axis=-1 )  )
	o = ( Conv2D( 128, (3, 3), activation='relu', padding='same', name='up8x_conv', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	fup8x = o

	o = (Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), name='upsample_8x', padding='same', activation='relu', data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([o,f2],axis=-1 ) )
	o = ( Conv2D( 64 , (3, 3), activation='relu', padding='same', name='up4x_conv', data_format=IMAGE_ORDERING ) )(o)
	o = ( BatchNormalization())(o)
	fup4x = o

	o = (Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2),  name='upsample_4x',padding='same', activation='relu', data_format=IMAGE_ORDERING))(o)
	o = ( concatenate([o,f1],axis=-1 ) )
	o = ( Conv2D( 64 , (3, 3), activation='relu', padding='same', name='up2x_conv', data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)
	fup2x = o

	out2x = Conv2D(n_classes, (1, 1) , activation='linear', padding='same', name='out2x', data_format=IMAGE_ORDERING )( fup2x )
	out4x = Conv2D(n_classes, (1, 1),   activation='linear', padding='same', name='out4x', data_format=IMAGE_ORDERING)( fup4x)
	out8x = Conv2D(n_classes, (1, 1),   activation='linear', padding='same', name='out8x', data_format=IMAGE_ORDERING)( fup8x)

	x4x = UpSampling2D( (2, 2), data_format=IMAGE_ORDERING)(out8x)
	eadd4x = Add()([x4x, out4x])

	x2x = UpSampling2D((2, 2), data_format=IMAGE_ORDERING)(eadd4x)
	eadd2x = Add()([x2x, out2x])

	model = Model(inputs=img_input, outputs=eadd2x)

	if frozenlayers:
	# freze front-end layers comes from vgg16
		vggLayerNames = [ 'block1_conv1', 'block1_conv2', 'block1_pool',
						  'block2_conv1', 'block2_conv2', 'block2_pool',
						  'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool',
						  #'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool'
						  ]

		for layer in model.layers:
			if layer.name in vggLayerNames:
				layer.trainable = False

	#sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9)
	adam = Adam(lr=1e-4)
	model.compile(optimizer=adam, loss=euclidean_loss, metrics=["accuracy"])

	return model



def VGGUnetV3( n_classes , input_height=512, input_width=512 , mode = 'train'):

	xmodel = VGGUnetV2(n_classes, input_height, input_width, mode)
	_, h, w, c = xmodel.output_shape
	reshapeOut = Reshape((h*w, c))(xmodel.output)

	newModel = Model(inputs=xmodel.input, outputs=reshapeOut)

	sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9)
	newModel.compile(optimizer=sgd, loss=euclidean_loss, metrics=["accuracy"])

	return newModel


