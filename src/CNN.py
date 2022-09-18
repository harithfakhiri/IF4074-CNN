import numpy as np

class CNNN():
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