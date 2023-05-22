import numpy as np
import mnist
from three_layer_net import ThreeLayerNet

X_train = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y_train = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz") 

idx = 0
dimension = 784
M = 50
C = 10
TRAIN_SIZE = 60000
BATCH_SIZE = 3

def init_network(X, Y) :
	batch = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
	x = np.empty([BATCH_SIZE, dimension])
	for i in range(BATCH_SIZE):
		x[i] = X[batch[i]].flatten()
	x = x / 255.0
	t = Y[batch]
	return x, t

network = ThreeLayerNet(input_size=dimension, hidden_size=M, output_size=C)

(X, T) = init_network(X_train, Y_train)

grad_numerical = network.numerical_gradient(X, T)
grad_backprop = network.gradient(X, T)

for key in grad_numerical.keys() :
	diff = np.abs(grad_numerical[key] - grad_backprop[key])
	print(key + ":" + str(diff))

