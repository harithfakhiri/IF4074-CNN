import numpy as np

class DenseLayer():
    def __init__(self, n_units, activation_function, n_inputs=None,):  
        self.bias = np.zeros((1, n_units))
        self.activation_function = activation_function
        self.n_units = n_units
        
        self.n_inputs = n_inputs

    def calc_activation_func(self, X):
        if (self.activation_function.lower() == "sigmoid"):
            out = []
            for i in range (len(X)):
                out.append(float(1/(1+np.exp(-X[i]))))
            return out
        elif (self.activation_function.lower() == "relu"):
            return np.maximum(0, X)

    def forward(self, inputs):
        output = []
        self.input = np.concatenate(([], inputs.flatten()))
        #inisiasi weight
        self.weight: np.array = [np.random.uniform(-1, 1, size=len(self.input)) for i in range(self.n_units)]
        self.deltaW = np.zeros((self.n_units))

        for i in range(self.n_units):
            # dot product
            output.append(np.dot(self.input, self.weight[i]))
        # Call activation function
        self.output = self.calc_activation_func(output)
        return self.output
    
    # Reset for error
    def _reset_error(self):
        self.deltaW = np.zeros((self.n_units))


    # Update weights and bias
    def update_weights(self, learning_rate, momentum):
        # Update weight formula = w  + momentum * w + learning_rate * errors * output
        # Update bias formula = bias + momentum * bias + learning_rate * errors
        for i in range(self.n_units):
            self.weight[i] = self.weight[i] - ((momentum * self.weight[i]) + (learning_rate * self.deltaW[i] * self.input))

        self.bias = self.bias - ((momentum * self.bias) + (learning_rate * self.deltaW))

        self._reset_error()
    
    # Update weight in the current layers based on previous errors and calculate error for the current network
    def backward(self, prev_errors):
        derivative_values = np.array([])
        for x in self.output:
            derivative_values = np.append(derivative_values, self.get_derivative(self.activation_function, x))
        self.deltaW += np.multiply(derivative_values, prev_errors)
        # weight matrix representation: row for output, column for input
        dE = np.matmul(prev_errors, self.weight)

        return dE
    
    def get_derivative(self, activation, x):
        if activation == 'sigmoid':
            return x * (1-x)
        elif activation == 'relu':
            if x >= 0:
                return 1
            else:
                return 0
        else:
            raise Exception("Undefined activation function")

# matrix = np.array([[[1,7,-2],[11,1,23],[2,2,2]],[[1,5,2],[10,-1,20],[4,2,4]],[[6,7,8],[12,-4,6],[8,2,6]]])

# dense = DenseLayer(2, 'RELU')

# output = dense.forward(matrix)

