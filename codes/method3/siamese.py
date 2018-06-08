from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten, MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import random
import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

INPUT_SHAPE = (227, 227, 3)
VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
session = tf.Session()
session.run(tf.global_variables_initializer())
class Siamese_Loader:
	def __init__(self, path, batch_size):
		self.batch_size = batch_size
		classes = glob.glob(path + '/*')
		self.image_set = [glob.glob(p + '/*') for p in classes]

	def get_batch(self):
		pairs = [np.zeros((self.batch_size,) + INPUT_SHAPE) for i in range(2)]
		targets = np.zeros((self.batch_size,))
		targets[self.batch_size // 2:] = 1
		for i in range(self.batch_size):
			if i >= self.batch_size // 2:
				images = random.choice(self.image_set)
				images = random.sample(images, 2)
				for j in range(2):
					image_string = tf.read_file(images[j])
					image_decoded = tf.image.decode_jpeg(image_string, channels=3)
					image_resized = tf.image.resize_images(image_decoded, [227, 227])
					image_centered = tf.subtract(image_resized, VGG_MEAN)
					pairs[j][i,:,:,:] = image_centered.eval(session=session)
			else:
				images = random.sample(self.image_set, 2)
				for j in range(2):
					image_string = tf.read_file(random.choice(images[j]))
					image_decoded = tf.image.decode_jpeg(image_string, channels=3)
					image_resized = tf.image.resize_images(image_decoded, [227, 227])
					image_centered = tf.subtract(image_resized, VGG_MEAN)
					pairs[j][i,:,:,:] = image_centered.eval(session=session)
		return pairs, targets

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

optimizer = Adam(0.00006)
siamese_net.compile(loss="mean_squared_error",optimizer=optimizer)

siamese_net.summary()

path = '/home/duanchx/DIP/training'
loader = Siamese_Loader(path, 8)

inputs = [np.zeros((8,) + INPUT_SHAPE) for i in range(2)]
targets = np.zeros((8,))

for i in range(90000):
	(inputs,targets) = loader.get_batch()
	loss = siamese_net.train_on_batch(inputs,targets)
	if i % 10 == 0:
		print("iteration {}, training loss: {:.2f},".format(i, loss))
	if i % 100 == 0:
		siamese_net.save('./parameter/epoch_%d.hdf5' % i)