import numpy as np
from sympy import symbols, diff
from lab5_utils import mean_squared_error, relu, ArtificialNeuralNetwork


def d_mse(y, y_hat):
	return -(y-y_hat)
def d_relu(x):
	sol = np.zeros_like(x)
	sol[x>=0] = 1
	return sol

# consider how you can use ArtificialNeuralNetwork.forward and ArtificialNeuralNetwork.at_layer to help with BP
def train(
    neural_network,
    training_inputs,
    training_labels,
    n_epochs,
    learning_rate=0.001
):
	losses = []
	#grab number of features
	m = training_inputs.shape[1]

	#loop through epochs
	for i in range(0, n_epochs):
		#grab y_hat = a and a/z_memory from forwards our inputs through NN
		a, a_memory, z_memory = neural_network.forward(training_inputs)
		#grab error and append to losses
		error = mean_squared_error(training_labels,a)
		losses.append(error)
		#grab derivated of error
		d_Al =d_mse(training_labels,a) 
		layers = np.array(neural_network.layers)
		#loop through each layer backwards to backpropegate
		for layer in range(len(neural_network.layers), 0 , -1):
			#grab a from previous layer
			A_l1 = a_memory[layer-1]
			#calculate partial derivate of z
			d_z = d_Al * d_relu(z_memory[layer])
			#calculate partial derivative of w
			d_w = np.dot(d_z, A_l1.T)
			#partial derivative of b
			d_beta = 1/m * (np.sum(d_z, axis = 1 , keepdims = True))

			
			w_l = np.array(layers[layer-1])
			#calculate new derivative of error
			d_Al = np.dot(w_l.T, d_z)

			
			temp = learning_rate*d_w
			#propegate adjusted values to network
			neural_network.layers[layer-1] = w_l - temp
			neural_network.biases[layer-1] = neural_network.biases[layer-1] - learning_rate * d_beta
		#grab new error after backpropegation is complete
		a, a_memory, z_memory = neural_network.forward(training_inputs)
		error = mean_squared_error(training_labels,a)
		#append to our losses to see progression
		losses.append(error)

	return losses


def extra_credit(x_train, y_train, x_test, y_test):
    #populate this dictionary with your extra credit experiments...
    results = {
        "architectures": [],
        "n_epochs": [],
        "learning_rate": [],
        "training_losses": [],
        "evaluation_mean_squared_error": []
    }

    return results