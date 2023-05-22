import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm

X_TRAIN = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y_TRAIN = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz") 
X_TEST = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
Y_TEST = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz") 

idx = 33003
np.random.seed(1)

# 学習データ数
TRAIN_SIZE = 60000
# バッチサイズ
BATCH_SIZE = 100
# 入力層のノード数(画像データの次元数)
dimension = 784
# 中間層のノード数
M = 128
# 出力層のノード数
C = 10

# 学習率
learning_rate = 0.5
# エポック数
EPOCH = TRAIN_SIZE // BATCH_SIZE

# パラメータ
# W_1 = np.empty([M, dimension])
# b_1 = np.empty([M, 1])
# W_2 = np.empty([C, M])
# b_2 = np.empty([C, 1])

# パラメータの初期値を設定
W_1 = np.random.normal(loc=0, scale=1/dimension, size=(M, dimension))
b_1 = np.random.normal(loc=0, scale=1/dimension, size=(M, 1))
W_2 = np.random.normal(loc=0, scale=1/M, size=(C, M))
b_2 = np.random.normal(loc=0, scale=1/M, size=(C, 1))


# バッチのインデックスを格納
batch = np.empty(BATCH_SIZE)
# 入力層への入力データ
X_1 = np.empty([dimension, BATCH_SIZE])
# 中間層への入力データ
U_1 = np.empty([M, BATCH_SIZE])
# 中間層からの出力データ
Z_1 = np.empty([M, BATCH_SIZE])
# 出力層への入力データ
U_2 = np.empty([C, BATCH_SIZE])
# 出力層からの入力データ
Y = np.empty([C, BATCH_SIZE])

grad_U_2 = np.empty([C, BATCH_SIZE])
grad_Z_1 = np.empty([M, BATCH_SIZE])
grad_W_2 = np.empty([C, M])
grad_b_2 = np.empty([C, 1])
grad_U_1 = np.empty([M, BATCH_SIZE])
grad_X_1 = np.empty([dimension, BATCH_SIZE])
grad_W_1 = np.empty([M, dimension])
grad_b_1 = np.empty([M, 1])




# 教師データ
T = np.empty([BATCH_SIZE])
T_one_hot = np.empty([C, BATCH_SIZE])

def one_hot_vector(y, t):
	for i in range(BATCH_SIZE):
		res = np.zeros(C)
		res[y[i]] = 1.0
		t[:,i] = res
	return 

def init_network() :
	global batch, T, T_one_hot
	batch = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
	T = Y_TRAIN[batch]
	one_hot_vector(T, T_one_hot)
	return 

def input_layer():
	global X_1
	for i in range(BATCH_SIZE):
		X_1[:,i] = X_TRAIN[batch[i]].flatten()
	
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
	return -np.sum(np.log(y[t, np.arange(BATCH_SIZE)]))/BATCH_SIZE

def output_layer(x):
	global Y, T
	for i in range(BATCH_SIZE):
		Y[:,i] = softmax_function(x[:,i])
	print(cross_entropy_error(Y, T))
	return 

def feed_forward():
	global U_1, Z_1, U_2, grad_U_2
	init_network()
	input_layer()
	U_1 = all_joined_layer(b_1, W_1, X_1)
	Z_1 = sigmoid_function(U_1)
	U_2 = all_joined_layer(b_2, W_2, Z_1)
	output_layer(U_2)


def feed_backward():
	global grad_U_2, grad_Z_1, grad_W_2, grad_b_2, grad_U_1, grad_X_1, grad_W_1, grad_b_1
	grad_U_2 = (Y - T_one_hot)/BATCH_SIZE
	grad_Z_1 = np.dot(W_2.T, grad_U_2)
	grad_W_2 = np.dot(grad_U_2, Z_1.T)
	grad_b_2 = np.sum(grad_U_2, axis = 1)
	grad_U_1 = np.multiply(np.multiply(grad_Z_1, Z_1), 1 - Z_1)
	grad_X_1 = np.dot(W_1.T, grad_U_1)
	grad_W_1 = np.dot(grad_U_1, X_1.T)
	grad_b_1 = np.sum(grad_U_1, axis = 1)
	grad_b_1 = grad_b_1.reshape(-1, 1)
	grad_b_2 = grad_b_2.reshape(-1, 1)
	# print(grad_b_2)
	# print(grad_U_2)

def params_update():
	global W_1, b_1, W_2, b_2, grad_W_2, grad_b_2, grad_W_1, grad_b_1
	W_1 -= learning_rate * grad_W_1
	b_1 -= learning_rate * grad_b_1
	W_2 -= learning_rate * grad_W_2
	b_2 -= learning_rate * grad_b_2



def main():
	for i in range(EPOCH): 
		feed_forward()
		feed_backward()
		params_update()
	
	np.save(
		"params_w_1",
		W_1
	)
	np.save(
		"params_b_1",
		b_1
	)
	np.save(
		"params_w_2",
		W_2
	)
	np.save(
		"params_b_2",
		b_2
	)
	return

main()









		
			




