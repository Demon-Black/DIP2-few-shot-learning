import tensorflow as tf
import numpy as np

f = np.load('f.npy')
f_test = np.load('f_testing.npy')

x1 = tf.placeholder('float', [None, 4096])
x2 = tf.placeholder('float', [None, 4096])
y = tf.placeholder('float', [None, 1])

def mlp(X1, X2, weights, bias):
	X = tf.abs(tf.subtract(x1, x2))
	X = tf.add(tf.matmul(X, weights['out']), bias['out'])
	X = tf.nn.sigmoid(X)
	return X

weight = {
    'out': tf.Variable(tf.random_normal([4096, 1]))
}

bias = {
    'out': tf.Variable(tf.random_normal([1]))
}

y_ = mlp(x1, x2, weight, bias)
predict = np.zeros((2500, 50, 10))

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	saver.restore(session, './p/iteraction99000')
	t = np.expand_dims(np.array([0]), axis=0)
	for i in range(2500):
		print i
		pairs1 = f_test[i, :].reshape(1, 4096)
		for j in range(50):
			for k in range(10):
				pairs2 = f[j, k, :].reshape(1, 4096)
				p = session.run(y_, feed_dict={x1: pairs1, x2: pairs2, y: t})
				predict[i, j, k] = p[0][0]

np.save('predict.npy', predict)
