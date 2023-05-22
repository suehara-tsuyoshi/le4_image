import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm

X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz") 
idx = 33003

# 学習データ数
SAMPLE_NUM = 60000
# 入力層のノード数(画像データの次元数)
dimension = 784
# 中間層のノード数
M = 32
# 出力層のノード数
C = 10

X_1 = np.empty(dimension)
W_1 = np.empty([dimension, M])
b_1 = np.empty(M)
W_2 = np.empty([M, C])
b_2 = np.empty(C)

def init_network() :
	global W_1, b_1, W_2, b_2
	np.random.seed(1)
	W_1 = np.random.normal(loc=0, scale=1/dimension, size=(dimension, M))
	b_1 = np.random.normal(loc=0, scale=1/dimension, size=M)
	W_2 = np.random.normal(loc=0, scale=1/M, size=(M, C))
	b_2 = np.random.normal(loc=0, scale=1/M, size=C)
	return 

def input_layer():
	global X_1
	X_1 = X[idx].flatten()
	X_1 = X_1 / 255.0
	return 

def all_joined_layer(b, w, x):
	return np.dot(x, w) + b

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
	print(y)
	print(np.argmax(y))
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







		
			




