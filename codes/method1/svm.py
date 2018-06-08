from sklearn.neighbors import NearestNeighbors  
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

p = np.load("p_softmax.npy")
classifier = SVC()
x = p[:, :8, :].reshape((-1, 1000))
y = np.array([])
for i in range(50):
	for j in range(8):
		y = np.append(y, np.array([i + 1]))


classifier.fit(x, y)
joblib.dump(classifier, 'svm_p_softmax.pkl')

for i in range(50):
	for j in range(2):
		x = p[i, 8 + j, :]
		print(classifier.predict(x))
