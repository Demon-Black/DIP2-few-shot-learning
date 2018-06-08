from alexnet import AlexNet
import tensorflow as tf
import numpy as np
import glob

# print glob.glob('./training/*')

x = tf.placeholder(tf.float32, shape=[None,227,227,3], name='X')
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, 1000, [])
# p = tf.nn.softmax(model.fc8)

f = np.zeros([2500, 4096])

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	model.load_initial_weights(session)
	VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
	for d, i in enumerate(sorted(glob.glob('./testing/*'))):
		image_string = tf.read_file(i)
		image_decoded = tf.image.decode_jpeg(image_string, channels=3)
		image_resized = tf.image.resize_images(image_decoded, [227, 227])
		image_centered = tf.subtract(image_resized, VGG_MEAN)
		# RGB -> BGR
		image_bgr = image_centered[:, :, ::-1]
		image_bgr = image_bgr.eval(session=session)	
		feature = session.run(model.fc7, feed_dict={x: np.expand_dims(image_bgr, axis=0), keep_prob:1.0})
		# feature = session.run(model.fc8, feed_dict={x: np.expand_dims(image_bgr, axis=0), keep_prob:1.0})
		# feature = session.run(p, feed_dict={x: np.expand_dims(image_bgr, axis=0), keep_prob:1.0})
		f[d, :] = feature
	print f.shape
	np.save('f_testing.npy', f)
