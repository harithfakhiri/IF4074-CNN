import numpy as np

class DenseLayer():
    def __init__(self, n_units, activation_function, n_inputs=None,):  
        self.bias = np.zeros((1, n_units))
        self.activation_function = activation_function
        self.n_units = n_units
        
        self.n_inputs = n_inputs
        
        if n_inputs:
            self.weight = 0.10 * np.random.randn(n_inputs, n_units)


    def calc_activation_func(self, X):
        if (self.activation_function.lower() == "sigmoid"):
            return float(1/(1+np.exp(-X)))
        elif (self.activation_function.lower() == "relu"):
            return np.maximum(0, X)

    def forward(self, inputs):
        if (not self.n_inputs):
            self.n_inputs = len(inputs)
            self.inputs = inputs
            self.weight = 0.10 * np.random.randint(-10, 10, size=(self.n_inputs, self.n_units))

        print('=========WEIGHT=========')
        print(self.weight)
        output = np.dot(inputs, self.weight) + self.bias
        print('=========DOT PRODUCT=========')
        print(output)
        self.output = self.calc_activation_func(output)
        print('=========OUTPUT==========')
        print(self.output)

        return self.output

matrix = np.array([[[1,7,-2],[11,1,23],[2,2,2]],[[1,5,2],[10,-1,20],[4,2,4]],[[6,7,8],[12,-4,6],[8,2,6]]])

dense = DenseLayer(2, 'RELU')

output = dense.forward(matrix)

