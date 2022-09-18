# Library for plotting the images and the loss function
import matplotlib.pyplot as plt

# We import the data set from tensorflow and build the model there

#others
import os
import numpy as np
from numpy import asarray
from PIL import Image

class Utils():
    def __init__(self):
        self.root = os.path.dirname(os.path.abspath(__file__)) +'/../dataset'
        self.path_dogs = self.createPath("dogs")
        self.path_cats = self.createPath("cats")

    # change the working directory to the path where the images are located
    def createPath(self, animal):
        return self.root + '/' + animal

    def loadTrainDataSet(self):
        os.chdir(self.path_dogs)

        # this list holds all the image filename
        dogs = []
        
        # creates a ScandirIterator aliased as files
        with os.scandir(self.path_dogs) as files:
        # loops through each file in the directory
            for file in files:
                if file.name.endswith('.jpg'):
                # adds only the image files to the sceneries list
                    dogs.append(file.name)
        
        os.chdir(self.path_cats)
        
        # this list holds all the image filename
        cats = []
        
        # creates a ScandirIterator aliased as files
        with os.scandir(self.path_cats) as files:
        # loops through each file in the directory
            for file in files:
                if file.name.endswith('.jpg'):
                # adds only the image files to the sceneries list
                    cats.append(file.name)

        return dogs, cats

    def createMatrix(self, animal_arr, idx, animal):
        if (animal.lower() =='dogs'):
            image = Image.open(self.path_dogs+'/'+animal_arr[0][idx])
            data = asarray(image)
        elif (animal.lower() =='cats'):
            image = Image.open(self.path_cats+'/'+animal_arr[1][idx])
            data = asarray(image)

        squared = self.squaredPadding(data)
        red = self.createOneColor(squared, 0)
        green = self.createOneColor(squared, 1)
        blue = self.createOneColor(squared, 2)
        split_matrix = np.array([red,green,blue])
        # print(split_matrix)
        # print(split_matrix.shape)
        return split_matrix

    # def createRedOnly(animal_arr):
    #     for i in range(len(animal_arr)):
    #         for j in range(len(animal_arr[0])):
    #             for k in range(len(animal_arr[0][0])):
    #                 if(k!=0):
    #                     animal_arr[i][j][k] = 0
    #     return animal_arr

    def loadAllData(self, animal_arr):
        #load dogs
        total_matrix = []
        for i in range(len(animal_arr[0])):
            total_matrix.append(self.createMatrix(animal_arr, i, "dogs"))

        #load cats
        for i in range(len(animal_arr[1])):
            total_matrix.append(self.createMatrix(animal_arr, i, "cats"))
        #labeling array

        # 0 == dogs
        # 1 == cats
        label_array = np.zeros(len(total_matrix))
        for i in range(len(label_array)):
            if (i < len(animal_arr[0])):
                label_array[i] = 0
            else:
                label_array[i] = 1
        
        return total_matrix, label_array
    
    def createOneColor(self, color_one,idx):
        height_px = len(color_one)
        width_px = len(color_one[0])
        red = np.zeros((height_px, width_px), dtype=int)
        for i in range(height_px):
            for j in range(width_px):
                red[i][j] = color_one[i][j][idx]
        return red
        
    def squaredPadding(self, RGB_Matrix):
        height, width = RGB_Matrix.shape[0], RGB_Matrix.shape[1]
        matrix = np.zeros((height,width,3),dtype=int)
        if (height>width):
            new_height = height
            new_width = height
        else : 
            new_height = width
            new_width = width
        padding_h = int((new_height - height)/2)
        padding_w = int((new_width - width)/2)
        # print(padding_h, padding_w)
        # print(new_height, new_width)
        matrix = np.zeros((new_height,new_width,3),dtype=int)
        #inserting img to zeros
        matrix[padding_h:height + padding_h, padding_w:width + padding_w] = RGB_Matrix
        # print(matrix)
        return matrix
        

    # dogs = loadTrainDataSet(root, 'dogs')
    # print(dogs)
    # cats = loadTrainDataSet(root, 'cats')
    # createMatrix(createPath(root,'dogs'), dogs, 9)