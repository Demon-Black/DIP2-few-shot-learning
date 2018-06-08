import tensorflow as tf
import numpy as np
import random

x1 = tf.placeholder('float', [None, 4096])
x2 = tf.placeholder('float', [None, 4096])
y = tf.placeholder('float', [None, 1])

f = np.load('f.npy')
def get_batch(batch_size):
	pairs = [np.zeros((batch_size,) + (4096,)) for i in range(2)]
	targets = np.zeros((batch_size,))
	for i in range(batch_size):
		if random.randint(0, 1) == 1:
			c = random.randint(0, 49)
			n1 = random.randint(0, 9)
			n2 = random.randint(0, 9)
			while n1 == n2:
				n2 = random.randint(0, 9)
			pairs[0][i, :] = f[c, n1, :]
			pairs[1][i, :] = f[c, n2, :]
			targets[i] = 1
		else:
			c1 = random.randint(0, 49)
			c2 = random.randint(0, 49)
			while c1 == c2:
				c2 = random.randint(0, 49)
			n1 = random.randint(0, 9)
			n2 = random.randint(0, 9)
			pairs[0][i, :] = f[c1, n1, :]
			pairs[1][i, :] = f[c2, n2, :]
			targets[i] = 0
	return pairs, targets.reshape(32, 1)

def mlp(X1, X2, weights, bias):
	X = tf.abs(tf.subtract(x1, x2))
	# X = tf.add(tf.matmul(X, weights['l1']), bias['l1'])
	# X = tf.nn.relu(X)
	# X = tf.add(tf.matmul(X, weights['l2']), bias['l2'])
	# X = tf.nn.relu(X)
	# X = tf.add(tf.matmul(X, weights['l3']), bias['l3'])
	# X = tf.nn.relu(X)
	X = tf.add(tf.matmul(X, weights['out']), bias['out'])
	# X = tf.nn.softmax(tf.nn.relu(X))
	X = tf.nn.sigmoid(X)
	return X

weight = {
    # 'l1': tf.Variable(tf.random_normal([4096, 256])),
    # 'l2': tf.Variable(tf.random_normal([512, 256])), 
    # 'l3': tf.Variable(tf.random_normal([1024, 512])), 
    'out': tf.Variable(tf.random_normal([4096, 1]))
}

bias = {
    # 'l1': tf.Variable(tf.random_normal([256])),
    # 'l2': tf.Variable(tf.random_normal([256])), 
    # 'l3': tf.Variable(tf.random_normal([512])), 
    'out': tf.Variable(tf.random_normal([1]))
}

y_ = mlp(x1, x2, weight, bias)
cost = tf.reduce_sum(tf.square(y - y_))
optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)

min_c = 32

with tf.Session() as session:
	session.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	for epoch in range(100000):
		pairs, targets = get_batch(32)
		_, c = session.run([optimizer, cost], feed_dict={x1: pairs[0], x2: pairs[1], y: targets})
		if epoch % 50 == 0:
			print c
		if c < min_c:
			min_c = c
			saver.save(session, './p/iteraction%d' % epoch)
		if epoch % 1000 == 0:
			saver.save(session, './p/iteraction%d' % epoch)



