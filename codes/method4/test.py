import numpy as np
from sklearn.neighbors import NearestNeighbors  

f = np.load('f.npy')
centers = np.array([np.mean(f[i, :, :], axis=0) for i in range(f.shape[0])])
print centers.shape
knn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(centers)
# for i in range(50):
# 	distances, indices = knn.kneighbors(f[i, 8:, :])
# 	print indices + 1

f_test = np.load('f_testing.npy')
distances, indices = knn.kneighbors(f_test)
print indices.shape
with open('result-c.txt', 'w') as file:
	for i in range(2500):
		file.write(str(indices[i][0] + 1) + '\n')