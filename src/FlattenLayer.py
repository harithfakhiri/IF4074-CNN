class FlattenLayer:
    def init(self):
        pass

    def forward(self, inputs):
        flattened_map = inputs.flatten()
        return flattened_map