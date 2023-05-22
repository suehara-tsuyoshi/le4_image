import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm

X_TEST = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
Y_TEST = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz") 
idx = 100

# 学習データ数
SAMPLE_NUM = 60000
# 入力層のノード数(画像データの次元数)
dimension = 784
# 中間層のノード数
M = 128
# 出力層のノード数
C = 10

X_1 = np.empty(dimension)
W_1 = np.empty([M, dimension])
b_1 = np.empty([M, 1])
W_2 = np.empty([C, M])
b_2 = np.empty([C, 1])

def init_network() :
	global W_1, b_1, W_2, b_2
	W_1 = np.load(file="params_w_1.npy")
	b_1 = np.load(file="params_b_1.npy")
	W_2 = np.load(file="params_w_2.npy")
	b_2 = np.load(file="params_b_2.npy")
	return 

def input_layer():
	global X_1
	X_1 = X_TEST[idx].flatten()
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

def output_layer(x):
	y = softmax_function(x)
	# print(y)
	# print(Y_TEST[idx])
	return 

def main():
	global Y_1, X_2, Y_2
	init_network()
	input_layer()
	Y_1 = all_joined_layer(b_1, W_1, X_1)
	print(X_1.shape)
	X_2 = sigmoid_function(Y_1)
	Y_2 = all_joined_layer(b_2, W_2, X_2)
	output_layer(Y_2)
	return

main()







		
			




