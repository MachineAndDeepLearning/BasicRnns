import numpy as np
import string


def init_weight(Mi, Mo):
	return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


def all_parity_pairs(nbit):
	N = 2 ** nbit
	remainder = 100 - (N % 100)
	Ntotal = N + remainder
	X = np.zeros((Ntotal, nbit))
	Y = np.zeros(Ntotal)

	for ii in range(Ntotal):
		i = ii % N

		for j in range(nbit):
			if i % (2 ** (j + 1)) != 0:
				i -= 2 ** j
				X[ii, j] = 1
		Y[ii] = X[ii].sum() % 2
	return X, Y


def all_parity_pairs_with_sequence_labels(nbit):
	X, Y = all_parity_pairs(nbit)
	N, t = X.shape

	# we want every time step to have a label
	Y_t = np.zeros(X.shape, dtype=np.int32)
	for n in range(N):
		ones_count = 0
		for i in range(t):
			if X[n, i] == 1:
				ones_count += 1
			if ones_count % 2 == 1:
				Y_t[n, i] = 1

	X = X.reshape(N, t, 1).astype(np.float32)
	return X, Y_t


def remove_punctuation(s):
	translator = str.maketrans('', '', string.punctuation)
	return s.translate(translator)
