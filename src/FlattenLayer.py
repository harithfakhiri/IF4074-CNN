class FlattenLayer:
    def init(self):
        pass

    def forward(self, inputs):
        self.C, self.W, self.H = inputs.shape
        flattened_map = inputs.flatten()
        return flattened_map

    def backward(self, prev_errors):
        return prev_errors.reshape(self.C, self.W, self.H)

    def update_weights(self, learning_rate, momentum):
        pass