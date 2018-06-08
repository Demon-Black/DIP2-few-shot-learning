import numpy as np

p = np.load('predict.npy')

def sum_probability(i):
	sums = np.array([np.sum(p[i, m, :]) for m in range(50)])
	return sums.argmax() + 1

def max_probability(i):
	max_p = np.array([np.max(p[i, m, :]) for m in range(50)])
	return max_p.argmax() + 1

def majority_voting(i):
	voting = np.array([np.sum(p[i, m, :] > 0.5) for m in range(50)])
	return voting.argmax() + 1


with open('result.txt', 'w') as f:
	for i in range(2500):
		f.write(str(sum_probability(i)) + '\n')
