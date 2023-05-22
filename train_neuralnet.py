import numpy as np
import mnist
from three_layer_net import ThreeLayerNet
from optimizer import * 

x_train = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
t_train = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz") 
x_test = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
t_test = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz") 

idx = 0
dimension = 784
M = 128
C = 10
test_size = 10000
train_size = 60000
batch_size = 100
learning_rate = 0.1
epochs = train_size // batch_size

def init_test_data(X, T):
	batch = np.random.choice(test_size, batch_size)
	x = np.empty([test_size, dimension])
	for i in range(test_size):
		x[i] = X[i].flatten()
	x = x / 255.0
	t = T
	return x, t

def init_network(X, T) :
	batch = np.random.choice(train_size, batch_size)
	x = np.empty([batch_size, dimension])
	for i in range(batch_size):
		x[i] = X[batch[i]].flatten()
	x = x / 255.0
	t = T[batch]
	return x, t

network = ThreeLayerNet(input_size=dimension, hidden_size=M, output_size=C)
optimizer = AdaGrad()
(x_test_batch, t_test_batch) = init_test_data(x_test, t_test)

for i in range(10*epochs) :
	(x_batch, t_batch) = init_network(x_train, t_train)
	grads = network.gradient(x_batch, t_batch)
	params = network.params
	optimizer.update(params, grads)
	
	loss = network.loss(x_batch, t_batch)
	if i % epochs == 0 :
		test_acc = network.accuracy(x_test_batch, t_test_batch)
		print(test_acc)

	

