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

    def backward(self, dE):
        delta_E = dE
        for i in reversed(range(len(self.layers))):
            # print("index backward: ", i)
            delta_E = self.layers[i].backward(delta_E)
        return delta_E

    def predict(self, features, target, batch_size, epochs, learning_rate, momentum=1):
        out = np.array([])
        y_target = np.array([])
        for i in range(epochs):
            
            print("Epoch:", i+1, end=', ')
            sum_loss = 0
            for j in range(len(features)):
                # feat_per_batch = len(features)//batch_size  
                # curr_index = j * feat_per_batch
                # # curr_index = (batch_size * i + j) % len(features) 
                # # Feed forward
                # # print(features[curr_index:curr_index+feat_per_batch][0])
                # for k in range(curr_index, curr_index+feat_per_batch):
                #     result = self.forward(features[k])
                #     curr_target = target[k]
                #     curr_output = result[0]
                #     delta_E = np.array([curr_target - curr_output])*-1
                #     delta_E = self.backward(delta_E)
                #     sum_loss += self.calculate_loss(curr_target, curr_output)
                #     out = np.rint(np.append(out, curr_output))
                #     y_target = np.append(y_target, curr_target)
                result = self.forward(features[j])
                print("masuk forward", j )
                curr_target = target[j]
                curr_output = result[0]
                delta_E = np.array([curr_target - curr_output])*-1
                delta_E = self.backward(delta_E)
                print("masuk backward", j)
                sum_loss += self.calculate_loss(curr_target, curr_output)
                out = np.rint(np.append(out, curr_output))
                y_target = np.append(y_target, curr_target)
                if(j+1%batch_size == 0):
                    for i in reversed(range(len(self.layers))):
                        self.layers[i].update_weights(learning_rate, momentum)
        
            avg_loss = sum_loss/len(features)
            print('Loss: ', avg_loss, end=', ')
            # print("y target", y_target)
            # print("output", out)
            print('Accuracy: ', metrics.accuracy_score(y_target, out))