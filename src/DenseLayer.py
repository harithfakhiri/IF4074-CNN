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
        self.input = np.concatenate(([1], inputs.flatten()))
        self.weight: np.array = [np.random.uniform(-1, 1, size=len(self.input)) for i in range(self.n_units)]

        for i in range(self.n_units):
            # Sum Product
            output.append(np.dot(self.input, self.weight[i]))
        self.output = self.calc_activation_func(output)
        return self.output

# matrix = np.array([[[1,7,-2],[11,1,23],[2,2,2]],[[1,5,2],[10,-1,20],[4,2,4]],[[6,7,8],[12,-4,6],[8,2,6]]])

# dense = DenseLayer(2, 'RELU')

# output = dense.forward(matrix)

