import numpy as np

class ConvolutionalStage():
    def __init__(self, filter_size, num_filter,  num_channel, stride=1, padding=0):
        self.num_channel = num_channel
        self.padding = padding
        self.filter_size = filter_size
        self.num_filter = num_filter
        self.stride = stride
        self.bias = np.zeros((num_filter))
        self.kernel = np.random.randn(self.num_filter, self.num_channel, self.filter_size, self.filter_size)
        self._dw = np.zeros((num_filter, num_channel, filter_size, filter_size))
        self._db = np.zeros((num_filter))
        

    def iterate_regions(self, image, n, width, height): 
        for i in range(height-(n-1)):
            for j in range(width-(n-1)):
                im_region = image[i:(i+n), j:(j+n)]
                yield im_region, i, j

    def zero_padding(self, image, new_width, new_height):
        width, height = image.shape
        output = np.zeros((new_width, new_height))
        output[self.padding:width + self.padding, self.padding:height + self.padding] = image
        return output
    
    def getOutputSize(self, width, height):
        out_width = int((width - self.filter_size)/self.stride + 1)
        out_heigth = int((height - self.filter_size)/self.stride + 1)

        return out_width, out_heigth
        
    def forward(self, inputs):
        channel = inputs.shape[0]
        width = inputs.shape[1]+2*self.padding
        height = inputs.shape[2]+2*self.padding
        
        out_width, out_heigth = self.getOutputSize(width, height)

        self.inputs = np.zeros((channel, width, height))
        feature_maps = np.zeros((self.num_filter, out_width, out_heigth))
        for c in range(channel):
            self.inputs[c, :, :] = self.zero_padding(inputs[c, :, :], width, height)
            # print(self.inputs[c])
        for f in range(self.num_filter):
            for i in range (out_width):
                for j in range(out_heigth):
                    feature_maps[f, i, j] = np.sum(
                        self.inputs[:, i:i+self.filter_size, j:j+self.filter_size] * self.kernel[f, :, :, :]) + self.bias[f]
        return feature_maps

    # Update weight when 1 batch is finished
    def updatekernel(self,learning_rate, momentum):
        self.kernel -= learning_rate * self._dw
        self.bias -= learning_rate * self._db

        self.error_back_zeros()
    
    def error_back_zeros(self):
        self._dw = np.zeros((self.num_filter, self.num_channel, self.filter_size, self.filter_size))
        self._db = np.zeros((self.num_filter))
        
    def backward(self, prev_errors):
        C, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        dw = np.zeros(self.kernel.shape)
        db = np.zeros(self.bias.shape)

        F, W, H = prev_errors.shape
        for f in range(F):
            for w in range(W):
                for h in range(H):
                    dw[f,:,:,:]+=prev_errors[f,w,h]*self.inputs[:,w:w+self.filter_size,h:h+self.filter_size]
                    dx[:,w:w+self.filter_size,h:h+self.filter_size]+=prev_errors[f,w,h]*self.kernel[f,:,:,:]

        for f in range(F):
            db[f] = np.sum(prev_errors[f, :, :])

        self._dw += dw
        self._db += db
        return dx

class PoolingStage():
    def __init__(self, filter_size, stride, isMax):
        self.filter_size = filter_size
        self.stride = stride
        self.isMax = isMax
    
    def forward(self, inputs):
        self.inputs = inputs
        chanel = inputs.shape[0]
        out_width = int((inputs.shape[1] - self.filter_size) / self.stride) + 1
        out_height = int((inputs.shape[2] - self.filter_size) / self.stride) + 1

        output = np.zeros([chanel, out_width, out_height], dtype=np.double)
        for c in range(chanel):
            for i in range(out_width):
                for j in range(out_height):
                    if (self.isMax):
                        output[c, i, j] = self.findMax(inputs[c, i:i+self.filter_size, j:j+self.filter_size])
                    else:
                        output[c, i, j] = self.findAvg(inputs[c, i:i+self.filter_size, j:j+self.filter_size])

        return output

    def findMax(self, inputs):
        max = np.max(inputs)
        return max

    def findAvg(self,inputs):
        avg = '%.3f' % np.average(inputs)
        return avg

    def backward(self, prev_errors):
        F, W, H = self.inputs.shape
        dx = np.zeros(self.inputs.shape)
        for f in range(0, F):
            for w in range(0, W, self.filter_size):
                for h in range(0, H, self.filter_size):
                    st = np.argmax(self.inputs[f, w:w+self.filter_size, h:h+self.filter_size])
                    (idx, idy) = np.unravel_index(st, (self.filter_size, self.filter_size))
                    if ((w + idx) < W and (h+idy) < H):
                        dx[f, w+idx, h+idy] = prev_errors[f, int(w/self.filter_size) % prev_errors.shape[1], int(h/self.filter_size) % prev_errors.shape[2]]
        return dx
    



class DetectionStage():
    def __init__(self, activation_function):
        self.activation_function = activation_function
    
    def calc_activation_func(self, X):
        if (self.activation_function.lower() == "sigmoid"):
            return float(1/(1+np.exp(-X)))
        elif (self.activation_function.lower() == "relu"):
            return np.maximum(0, X)

    def forward(self, inputs):
        self.inputs = inputs
        chanel = inputs.shape[0]
        width = inputs.shape[1]
        height = inputs.shape[2]
        output = np.zeros([chanel, width, height], dtype=np.double)
        for c in range(chanel):
            output[c, :, :] = self.calc_activation_func(inputs[c, :, :])
        
        return output

    def backward(self, prev_errors):
        dx = prev_errors.copy()
        dx[self.inputs < 0] = 0
        return dx

class ConvolutionLayer():
    def __init__(self, filter_size, num_filter,  num_channel, isMax, act_func_detection, stride=1, padding=0):
        self.convolution_stage = ConvolutionalStage(filter_size, num_filter,  num_channel, stride, padding)
        self.detection_stage = DetectionStage(act_func_detection)
        self.pooling_stage = PoolingStage(filter_size, stride, isMax)
    
    def forward(self, inputs):
        feature_map = self.convolution_stage.forward(inputs)
        output_detection = self.detection_stage.forward(feature_map)
        output_pooling = self.pooling_stage.forward(output_detection)

        return output_pooling
    
    def backward(self, prev_errors):
        dx_pooling = self.pooling_stage.backward(prev_errors)
        dx_detection = self.detection_stage.backward(dx_pooling)
        dx_convo = self.convolution_stage.backward(dx_detection)

        return dx_convo
    
    def update_weights(self, learning_rate, momentum):
        self.convolution_stage.updatekernel(learning_rate, momentum)



# matrix = np.array([[[1,7,-2],[11,1,23],[2,2,2]],[[1,5,2],[10,-1,20],[4,2,4]],[[6,7,8],[12,-4,6],[8,2,6]]])

# convo = ConvolutionalStage(2, 2, 3)

# # feature_maps = convo.forward(matrix)

# pooling = PoolingStage(2,1,True)

# pooled_map = pooling.forward(matrix)

# detect = DetectionStage()

# output = detect.forward(matrix, "relu")

# print(output)

# print(feature_maps)

