from typing import OrderedDict
import numpy as np
from functions import *
from layers import *
from gradient import *

class ThreeLayerNet :
	def __init__(self, input_size, hidden_size, output_size) :
		self.params = {}
		self.params['W1'] = np.random.normal(loc=0, scale=1/input_size, size=(input_size, hidden_size))
		self.params['b1'] = np.random.normal(loc=0, scale=1/input_size, size=(hidden_size))
		self.params['W2'] = np.random.normal(loc=0, scale=1/hidden_size, size=(hidden_size, output_size))
		self.params['b2'] = np.random.normal(loc=0, scale=1/hidden_size, size=(output_size))

		self.layers = OrderedDict()
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['Sigmoid'] = Sigmoid()
		# self.layers['Relu'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

		self.lastlayer = SoftmaxWithLoss()
	
	def predict(self, x) :
		for layer in self.layers.values() :
			x = layer.forward(x)
		return x
	
	def loss(self, x, t) :
		y = self.predict(x)
		return self.lastlayer.forward(y, t)
	
	def accuracy(self, x, t) :
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		if t.ndim != 1 :
			t = np.argmax(t, axis=1)
		accuracy = np.sum(y==t) / float(x.shape[0])
		return accuracy
	
	def numerical_gradient(self, x, t) :
		loss_W = lambda W: self.loss(x, t)
		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

		return grads
	
	def gradient(self, x, t) :
		self.loss(x, t)

		dout = 1
		dout = self.lastlayer.backward(dout)
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers :
			dout = layer.backward(dout)
		
		grads = {}
		grads['W1'] = self.layers['Affine1'].dW
		grads['b1'] = self.layers['Affine1'].db
		grads['W2'] = self.layers['Affine2'].dW
		grads['b2'] = self.layers['Affine2'].db
		return grads
