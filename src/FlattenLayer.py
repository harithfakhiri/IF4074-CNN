class FlattenLayer:
    def init(self):
        pass

    def forward(self, inputs):
        self.C, self.W, self.H = inputs.shape
        flattened_map = inputs.flatten()
        return flattened_map