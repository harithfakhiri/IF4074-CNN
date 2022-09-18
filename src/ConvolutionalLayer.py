from turtle import forward
import numpy as np

class ConvolutionalStage():
    def __init__(self, filter_size, num_filter,  num_channel, stride=1, padding=0):
        self.num_channel = num_channel
        self.padding = padding
        self.filter_size = filter_size
        self.num_filter = num_filter
        self.stride = stride
        self.bias = np.zeros((num_filter))
        self.kernel = np.random.randint(
            1, 6, size=(self.num_filter, self.num_channel, self.filter_size, self.filter_size))
        

    def iterate_regions(self, image, n, width, height): #n adalah ukuran receptive/field dan matriks kernel
        #generates all possible 3*3 image regions using valid padding
        for i in range(height-(n-1)):
            for j in range(width-(n-1)):
                im_region = image[i:(i+n), j:(j+n)]
                print(im_region)
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
        print(self.kernel)
        for c in range(channel):
            self.inputs[c, :, :] = self.zero_padding(inputs[c, :, :], width, height)
            # print(self.inputs[c])
            for f in range(self.num_filter):
                for i in range (out_width):
                    for j in range(out_heigth):
                        feature_maps[f, i, j] = np.sum(
                            self.inputs[:, i:i+self.filter_size, j:j+self.filter_size] * self.kernel[f, :, :, :]) + self.bias[f]
        return feature_maps
    

class PoolingStage():
    def __init__(self, filter_size, stride, isMax):
        self.filter_size = filter_size
        self.stride = stride
        self.isMax = isMax
    
    def forward(self, inputs):
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


class DetectionStage():
    def __init__(self):
        pass
    
    def calc_activation_func(self, X):
        if (self.activation_function.lower() == "sigmoid"):
            return float(1/(1+np.exp(-X)))
        elif (self.activation_function.lower() == "relu"):
            return np.maximum(0, X)

    def forward(self, inputs, activation_function):
        self.activation_function = activation_function
        chanel = inputs.shape[0]
        width = inputs.shape[1]
        height = inputs.shape[2]
        output = np.zeros([chanel, width, height], dtype=np.double)
        for c in range(chanel):
            output[c, :, :] = self.calc_activation_func(inputs[c, :, :])
        
        return output



matrix = np.array([[[1,7,-2],[11,1,23],[2,2,2]],[[1,5,2],[10,-1,20],[4,2,4]],[[6,7,8],[12,-4,6],[8,2,6]]])

convo = ConvolutionalStage(2, 2, 3)

# feature_maps = convo.forward(matrix)

pooling = PoolingStage(2,1,True)

pooled_map = pooling.forward(matrix)

detect = DetectionStage()

output = detect.forward(matrix, "relu")

print(output)

# print(feature_maps)

