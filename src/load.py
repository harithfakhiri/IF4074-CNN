# Library for plotting the images and the loss function
import matplotlib.pyplot as plt

# We import the data set from tensorflow and build the model there

#others
import os
import numpy as np
from numpy import asarray
from PIL import Image

root = os.path.dirname(os.path.abspath(__file__)) +'/../dataset'

# change the working directory to the path where the images are located
def createPath(root, animal):
    return root + '/' + animal

def loadDataSet(root, animal):
    path_animal = createPath(root, animal)
    os.chdir(path_animal)

    # this list holds all the image filename
    animal = []

    # creates a ScandirIterator aliased as files
    with os.scandir(path_animal) as files:
    # loops through each file in the directory
        for file in files:
            if file.name.endswith('.jpg'):
            # adds only the image files to the sceneries list
                animal.append(file.name)
    return animal

def createMatrix(path,animal_arr,idx):
    print(animal_arr[idx])
    image = Image.open(path+'/'+animal_arr[idx])
    # convert image to numpy array
    # print(image)
    data = asarray(image)

    squared = squaredPadding(data)
    red = createOneColor(squared, 0)
    green = createOneColor(squared, 1)
    blue = createOneColor(squared, 2)
    split_matrix = np.array([red,green,blue])
    print(split_matrix)
    print(split_matrix.shape)
    return split_matrix

def createRedOnly(animal_arr):
    for i in range(len(animal_arr)):
        for j in range(len(animal_arr[0])):
            for k in range(len(animal_arr[0][0])):
                if(k!=0):
                    animal_arr[i][j][k] = 0
    return animal_arr

def createOneColor(color_one,idx):
    height_px = len(color_one)
    width_px = len(color_one[0])
    red = np.zeros((height_px, width_px), dtype=int)
    for i in range(height_px):
        for j in range(width_px):
            red[i][j] = color_one[i][j][idx]
    return red
    
def squaredPadding(RGB_Matrix):
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
    print(padding_h, padding_w)
    print(new_height, new_width)
    matrix = np.zeros((new_height,new_width,3),dtype=int)
    #inserting img to zeros
    matrix[padding_h:height + padding_h, padding_w:width + padding_w] = RGB_Matrix
    print(matrix)
    return matrix
    

dogs = loadDataSet(root, 'dogs')
print(dogs)
cats = loadDataSet(root, 'cats')
createMatrix(createPath(root,'dogs'), dogs, 9)