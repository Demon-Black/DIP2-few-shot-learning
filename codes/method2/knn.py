from sklearn.neighbors import NearestNeighbors  
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

X = np.load("fc7.npy")
labels = np.load("label.npy")
f = np.load("f.npy")
# f_testing = np.load("f_testing.npy")
knn = NearestNeighbors(n_neighbors=100, algorithm="auto").fit(X)
l = np.zeros([50, 10, 100])
l_testing = np.zeros([2500, 100])
for i in range(50):
	distances, indices = knn.kneighbors(f[i, :, :])
	likely = np.zeros(indices.shape)
	for m in range(indices.shape[0]):
		for n in range(indices.shape[1]):
			likely[m, n] = labels[int(indices[m, n])]
	l[i, :, :] = likely

for i in range(2500):
	distances, indices = knn.kneighbors(f_testing[i, :])
	likely = np.zeros(indices.shape)
	for m in range(indices.shape[0]):
		likely[m, n] = labels[int(indices[m])]
	l_testing[i, :] = likely[m, n]

# np.save("likely.npy", l)
# likely = np.load("likely.npy")
classifier = SVC()
x = l[:, :, :].reshape((-1, 100))
y = np.array([])
for i in range(50):
	for j in range(10):
		y = np.append(y, np.array([i + 1]))


classifier.fit(x, y)
joblib.dump(classifier, 'svm.pkl')



# for i in range(2500):
# 	x = f_testing[i, :]
# 	print(classifier.predict(x))

