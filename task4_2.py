import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm

X_TEST = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
Y_TEST = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz") 
np.random.seed(1)

# 学習データ数
TRAIN_SIZE = 10000
# バッチサイズ
BATCH_SIZE = 10
# 入力層のノード数(画像データの次元数)
dimension = 784
# 中間層のノード数
M = 128
# 出力層のノード数
C = 10


X_1 = np.empty([dimension, BATCH_SIZE])
W_1 = np.empty([M, dimension])
b_1 = np.empty([M, 1])
W_2 = np.empty([C, M])
b_2 = np.empty([C, 1])
batch = np.empty(BATCH_SIZE)
T = np.empty(BATCH_SIZE)

def init_network() :
	global W_1, b_1, W_2, b_2, batch, T
	W_1 = np.load(file="params_w_1.npy")
	b_1 = np.load(file="params_b_1.npy")
	W_2 = np.load(file="params_w_2.npy")
	b_2 = np.load(file="params_b_2.npy")
	batch = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
	T = Y_TEST[batch]
	return 

def input_layer():
	global X_1
	for i in range(BATCH_SIZE):
		X_1[:,i] = X_TEST[batch[i]].flatten()
	
	X_1 = X_1 / 255.0
	return 

def all_joined_layer(b, w, x):
	return np.dot(w, x) + b

def sigmoid_function(x):
	return 1 / (1 + np.exp(-x))

def middle_layer(x):
	return sigmoid_function(x)

def softmax_function(x):
	alpha = np.max(x)
	y = np.exp(x-alpha)
	return y / np.sum(y)

def cross_entropy_error(y, t):
	return -np.sum(np.log(y[np.arange(BATCH_SIZE), t]))/BATCH_SIZE

def output_layer(x):
	global T
	y = np.empty([BATCH_SIZE, C])
	for i in range(BATCH_SIZE):
		y[i] = softmax_function(x[:,i])
	print(np.argmax(y, axis=1))
	print(T)
	return 

def main():
	init_network()
	input_layer()
	Y_1 = all_joined_layer(b_1, W_1, X_1)
	X_2 = sigmoid_function(Y_1)
	Y_2 = all_joined_layer(b_2, W_2, X_2)
	output_layer(Y_2)
	return

main()









		
			




