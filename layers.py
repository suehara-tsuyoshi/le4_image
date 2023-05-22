import numpy as np
from functions import *

class Relu :
	def __init__(self):
		self.mask = None
	
	def forward(self, x) :
		self.mask = np.array(x<=0)
		out = x.copy()
		out[self.mask] = 0
		return out
	
	def backward(self, dout) :
		dout[self.mask] = 0
		dx = dout
		return dx

class Sigmoid :
	def __init__(self) :
		self.out = None
	
	def forward(self, x) :
		out = sigmoid(x)
		self.out = out
		return out

	def backward(self, dout) :
		dx = dout * (1 - self.out) * self.out
		return dx
	
class Affine :
	def __init__(self, W, b) :
		self.x = None
		self.W = W
		self.b = b
		self.dW = None
		self.db = None
	
	def forward(self, x) :
		self.x = x
		out = np.dot(self.x, self.W) + self.b
		return out

	def backward(self, dout) :
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout) 
		self.db = np.sum(dout, axis = 0)
		return dx

class SoftmaxWithLoss :
	def __init__(self) :
		self.loss = None
		self.y = None
		self.t = None
	
	def forward(self, x, t) :
		self.y = softmax(x)
		self.t = t
		self.loss = cross_entropy_error(self.y, self.t)
		return self.loss
	
	def backward(self, dout) :
		batch_size = self.y.shape[0]
		if self.y.size == self.t.size :
			dx = (self.y - self.t) / batch_size
		
		else :
			dx = self.y.copy()
			dx[np.arange(batch_size), self.t] -= 1		
			dx = dx / batch_size

		return dx
	


