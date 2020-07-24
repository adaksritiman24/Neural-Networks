import numpy as np

np.random.seed(1)
#this is my first neural network from scratch in python (only numpy).
#it is a feed forward neural network with 2 hidden layers.

class FeedForwardNeuralNetwork_4L(object):
	def __init__(self, n_inputs, layer1, layer2, n_outputs):
		#declaring the weights of the network-----
		#weights from inputlayer to first hidden layer
		self.w0 = np.random.randn(n_inputs, layer1) - 1
		#weights from first hidden layer to second hidden layer
		self.w1 = np.random.randn(layer1, layer2) - 1
		#weights from second hidden layer to output layer
		self.w2 = np.random.randn(layer2, n_outputs) - 1

	def feedforward(self,X):
		#Pass input through our network and obtain an output and return it fron the function----
		#input value
		self.l0 = X
		#output from the first hidden layer after applying activation function on the weight multiplied input.
		self.l1 = self.sigmoid(np.dot(self.l0,self.w0))	
		#output fron the second hidden layer 
		self.l2 = self.sigmoid(np.dot(self.l1,self.w1))
		#output from the outputlayer
		self.l3 = self.sigmoid(np.dot(self.l2,self.w2))
		return self.l3

	def sigmoid(self,n, derivative = False):
		#this is the sigmoid activation function which takes any number and output a number between 0 and 1.
		if derivative==True:
			return n*(1-n)
		else:
			return 1/(1+np.exp(-n))	
			
	def backtrack(self,network_output,y,learning_rate):
		#Most important part : Training the network by (cost reduction by gradient descent)

		l3_error = y-network_output #this is actually the derivative of the cost function which is the squared error function.
		l3_delta = l3_error*self.sigmoid(self.l3, derivative = True) # the delta value is (derivative of cost function)*(derivative of the sigmoid function wrt. current layer output  )

		l2_error = l3_delta.dot(self.w2.T)#error of a hidden layer ,'l' is calculated by delta(l+1)*w(l)
		l2_delta = l2_error*self.sigmoid(self.l2,derivative = True)

		l1_error = l2_delta.dot(self.w1.T)
		l1_delta = l1_error*self.sigmoid(self.l1, derivative =True)

		#adjusting the weights
		#w(layer,l->l+1) is incremented by dot product of input to w(l){which is bacically d(input)/dw(l) } and the delta function for the layer, l+1.
		#learning rate keeps track of how fast we are reaching the minima of the cost function.
		self.w2+= self.l2.T.dot(l3_delta)*learning_rate
		self.w1+= self.l1.T.dot(l2_delta)*learning_rate
		self.w0+= self.l0.T.dot(l1_delta)*learning_rate

	def train(self,X,y, learning_rate):
		#feeding the input, X to the network , get an output ,followed by backtracking to adjust the weights.
		output = self.feedforward(X)
		self.backtrack(output,y,learning_rate)


#Dummy datas for learning:
#Input features (4 features)
X = np.array([[0,0,1,1],[1,0,1,1],[1,0,1,0],[1,1,1,0],[1,1,1,1]])
#Output (1 value)
y = np.array([[0],[1],[1],[0],[1]])

nn = FeedForwardNeuralNetwork_4L(4, 3, 2, 1)
#training the network 100,000 times with a learning rate of 0.2
for i in range(100000):
	nn.train(X, y, 0.2)
print("network_output :    Actual output:")
output= list(nn.feedforward(X))

#printing the network output alongside the desired output for the same input, 'X'.
for i in range(len(output)):
	print(output[i][0].round(3),"                 ",y[i][0])
