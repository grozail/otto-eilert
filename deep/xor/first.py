import numpy as np

def all_stuff():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]])
    
    Y = np.array([[0], [1], [1], [0]])
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    # variable initialization
    epoch = 10000
    learning_rate = 0.1
    input_layer_neurons = X.shape[1]
    hidden_layer_neurons = 2
    output_layer_neurons = 1
    
    # weights and bias initialization
    weights = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
    bias = np.random.uniform(size=(1, hidden_layer_neurons))
    weights_out = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
    bias_out = np.random.uniform(size=(1, output_layer_neurons))
    
    for i in range(epoch):
        # forward propagation
        hidden_layer_input = np.dot(X, weights)
        hidden_layer_input_biased = hidden_layer_input + bias
        hidden_layer_activations = sigmoid(hidden_layer_input_biased)
        output_layer_input = np.dot(hidden_layer_activations, weights_out)
        output_layer_input_biased = output_layer_input + bias_out
        output = sigmoid(output_layer_input_biased)
        
        # back propagation
        E = Y - output
        slope_output_layer = sigmoid_derivative(output)
        slope_hidden_layer = sigmoid_derivative(hidden_layer_activations)
        d_output = E * slope_output_layer
        E_at_hidden_layer = d_output.dot(weights_out.T)
        d_hidden_layer = E_at_hidden_layer * slope_hidden_layer
        weights_out += hidden_layer_activations.T.dot(d_output) * learning_rate
        bias_out += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        weights += X.T.dot(d_hidden_layer) * learning_rate
        bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
        if i % 1000 == 0:
            print("Output\n",output)
            print("Error\n", E)
            print("Weights\n", weights)

if __name__ == '__main__':
    all_stuff()