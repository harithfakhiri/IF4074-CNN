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

    def calculate_loss(self, target, output):
        return 0.5 * (target-output)**2

    def predict(self, features, target, batch_size, epochs, learning_rate, momentum=1):
        # out = np.array([])

        # for i in range(len(features)):
        #     result = self.forward(features[i])
        #     current_output = result[len(result)-1]
        #     out = np.rint(np.append(out, current_output))
        
        # return out
        out = np.array([])
        y_target = np.array([])
        for i in range(epochs):
            
            print("Epoch:", i+1, end=', ')
            sum_loss = 0
            for j in range(batch_size):
                curr_index = (batch_size * i + j) % len(features) 

                # Feed forward
                result = self.forward(features[curr_index])
                curr_target = target[curr_index]
                curr_output = result[len(result)-1]
                dE = np.array([curr_target - curr_output])*-1
                for i in reversed(range(len(self.layers))):
                    # print("index backward: ", i)
                    dE = self.layers[i].backward(dE)
                sum_loss += self.calculate_loss(curr_target, curr_output)
                out = np.rint(np.append(out, curr_output))
                y_target = np.append(y_target, curr_target)

            # Backward propagation
            # Output layer
            for i in reversed(range(len(self.layers))):
                self.layers[i].update_weights(learning_rate, momentum)
            avg_loss = sum_loss/batch_size
            print('Loss: ', avg_loss, end=', ')
            print('Accuracy: ', metrics.accuracy_score(y_target, out))