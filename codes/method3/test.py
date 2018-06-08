import glob
import numpy as np
from keras.models import load_model
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

INPUT_SHAPE = (227, 227, 3)
VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

def W_init(shape,name=None):
	values = rng.normal(loc=0,scale=1e-2,size=shape)
	return K.variable(values,name=name)

def b_init(shape,name=None):
	values=rng.normal(loc=0.5,scale=1e-2,size=shape)
	return K.variable(values,name=name)

input_shape = INPUT_SHAPE
left_input = Input(input_shape)
right_input = Input(input_shape)


convnet = Sequential()
convnet.add(Conv2D(32,(10,10),activation='relu',input_shape=input_shape,
				   kernel_initializer=W_init,kernel_regularizer=l2(2e-4)))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(64,(7,7),activation='relu',
				   kernel_regularizer=l2(2e-4),kernel_initializer=W_init,bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(64,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(128,(4,4),activation='relu',kernel_initializer=W_init,kernel_regularizer=l2(2e-4),bias_initializer=b_init))
convnet.add(Flatten())
convnet.add(Dense(4096,activation="sigmoid",kernel_regularizer=l2(1e-3),kernel_initializer=W_init,bias_initializer=b_init))
#encode each of the two inputs into a vector with the convnet
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)
#merge two encoded inputs with the l1 distance between them
L1_distance = lambda x: K.abs(x[0]-x[1])
both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
siamese_net = Model(input=[left_input,right_input],output=prediction)

session = tf.Session()
session.run(tf.global_variables_initializer())

siamese_net.load_weights('/home/duanchx/DIP/siamese/parameter/epoch_400.hdf5')

inputs = [np.zeros((1,) + INPUT_SHAPE) for i in range(2)]
classes = sorted(glob.glob('/home/duanchx/DIP/training/*'))
imageset = [sorted(glob.glob(p + '/*')) for p in classes]

predict = np.zeros((2500, 50, 10))

for i in range(1, 501):
	image_string = tf.read_file('/home/duanchx/DIP/testing/test_%04d.jpg' % i)
	image_decoded = tf.image.decode_jpeg(image_string, channels=3)
	image_resized = tf.image.resize_images(image_decoded, [227, 227])
	image_centered = tf.subtract(image_resized, VGG_MEAN)
	inputs[0][0,:,:,:] = image_centered.eval(session=session)
	for j in range(50):
		for k in range(10):
			image_string = tf.read_file(imageset[j][k])
			image_decoded = tf.image.decode_jpeg(image_string, channels=3)
			image_resized = tf.image.resize_images(image_decoded, [227, 227])
			image_centered = tf.subtract(image_resized, VGG_MEAN)
			inputs[1][0,:,:,:] = image_centered.eval(session=session)
			predict[i, j, k] = siamese_net.predict(inputs)[0][0]
			# print(predict[i, j, k])

np.save('predict.npy', predict)