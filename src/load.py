# Library for plotting the images and the loss function
import matplotlib.pyplot as plt

# We import the data set from tensorflow and build the model there

#others
import os
from numpy import asarray
from PIL import Image

root = 'C:/Users/Lenovo/Documents/semester 7/advanced ml/milestone a/IF4074-CNN/dataset'

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
    print(animal_arr)
    image = Image.open(path+'/'+animal_arr[idx])
    # convert image to numpy array
    print(image)
    data = asarray(image)
    print(type(data))
    # summarize shape
    print(data.shape)

    # create Pillow image
    image2 = Image.fromarray(data)
    print(type(image2))

    # summarize image details
    print(image2.mode)
    print(image2.size)
    print(data)
    print(len(data))
    print(len(data[0]))
    print(len(data[0][0]))

dogs = loadDataSet(root, 'dogs')
cats = loadDataSet(root, 'cats')
createMatrix(createPath(root,'dogs'), dogs, 0)