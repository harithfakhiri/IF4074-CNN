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

    def _forward(self,inputs):
        out = inputs.copy()
        result = [out]
        for layer in self.layers:
            out = layer.forward(out)
            result.append(out)
        return result

    def train(self, features, target, epochs, batch_size):
        out = np.array([])
        y_target = np.array([])

        for i in range(len(features)):
            result = self.forward(features[i])
            current_target = target[i]
            current_output = result[len(result)-1]
            out = np.rint(np.append(out, current_output))
            y_target = np.append(y_target, current_target)
        # for i in range(epochs):
            
        #     print("Epoch:", i+1, end=', ')
        #     sum_loss = 0
        #     for j in range(batch_size):
        #         curr_index = (batch_size * i + j) % len(features) 

        #         # Feed forward
        #         result = self._forward(features[curr_index])
        #         curr_target = target[curr_index]
        #         curr_output = result[len(result)-1][0]
        #         out = np.rint(np.append(out, curr_output))
        #         y_target = np.append(y_target, curr_target)
        
        return out