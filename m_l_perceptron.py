import numpy as np

np.random.seed(0)

#This is a multilayered feed forward neural network model.
#class initialization arguements:
# a. layers :- takes a list of elements where the list size will be the number of layers and list[i] value will be the number of nodes at layer,'i'.(compulsory)
#				Note, the number of nodes in first layer(first element in list) should be equal to the number of features in desired input in the training set.
#				also, the last element in list = number of features in the desired outputs in training.
# b. lr : learning rate of the model   (default = 0.2)
class MLP(object):
	def __init__(self,layers, lr = 0.2):
		self.lr = lr
		self.n = len(layers)
		self.weights = []
		self.z = []
		self.deltas = []
		self.bias = -1
		for i in range(self.n-1):
			self.weights.append(np.random.randn(layers[i],layers[i+1]) + self.bias)
	def feedforward(self,X):
		self.z.clear()
		self.z.append(X)
		for i in range(self.n-1):
			self.z.append(self.sigmoid(np.dot(self.z[i],self.weights[i])))
		return self.z[self.n-1]
	def sigmoid(self,s, deriv = False):
		if deriv ==True:
			return s*(1-s)
		else:
			return 1/(1+np.exp(-s))					 
	def backpropagate(self,n_out,y):
		self.deltas.clear()
		error = y- n_out
		for i in range(self.n-1,0,-1):
			if i== self.n-1:
				error = y-n_out
			else:
				error =	layer_delta.dot(self.weights[i].T)
			layer_delta = error*self.sigmoid(self.z[i],deriv = True)
			self.deltas.append(layer_delta)

		for i in range(self.n-1):	
			self.weights[i] += self.z[i].T.dot(self.deltas[self.n-2-i])*self.lr
	def train(self,X,y):
		output = self.feedforward(X)
		self.backpropagate(output,y)

