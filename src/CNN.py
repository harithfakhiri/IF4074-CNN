import numpy as np
from ConvolutionalLayer import ConvolutionLayer
from DenseLayer import DenseLayer
from sklearn import metrics

class CNN():
    def __init__(self, *layers):
        self.layers = []

        for l in layers:
            self.layers.append(l)
    
    def forward(self, inputs):
        output = inputs
        for i in range(len(self.layers)):
            output = self.layers[i].forward(output)
        return output


    def predict(self, features):
        out = np.array([])

        for i in range(len(features)):
            result = self.forward(features[i])
            current_output = result[len(result)-1]
            out = np.rint(np.append(out, current_output))
        
        return out