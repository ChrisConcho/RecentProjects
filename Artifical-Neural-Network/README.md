# Artificial Neural Network: Back Propegations

This is an ANN that uses back propegation to better fit the training data and make more accurate decisions on testing data. 

## Utilities
`ArtificialNeuralNetwork` is initialized with by passing in an array of layer sizes. The first layer must be the same 
size as the input. So if `training_inputs.shape == (n_features, n_samples)` then the first layer will have to be set to
`n_features`. Similarly the output must be equal to the size of the output which in this lab will always be 1. 

The class has 2 function associated with it:
* `forward` which computes the output, intermediate activation, and pre-activation values at each layer
* `at_layer` which computes the forward pass upto the specified layer.

## Tests

There are two tests, one is on randomly generated data points that have a large enough separation that the models 
should successfully learn a simple transformation that works and a synthetic dataset that represents a 3-D S shaped curve
as seen below:



![s-curve-regression-results.png](s-curve-regression-results.png)
