# Library for plotting the images and the loss function
import matplotlib.pyplot as plt

# We import the data set from tensorflow and build the model there

#others
import os
from numpy import asarray
from PIL import Image

root = os.path.dirname(os.path.abspath(__file__)) +'\..\dataset'
print(root)

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
    print("red only")
    red = createOneColor(data, 0)
    blue = createOneColor(data, 1)
    green = createOneColor(data, 2)
    # print(createRedOnly(data))
    print("red")
    print(red[0][0])
    print(red[0][1])
    print(red[0][2])
    print(red[1][0])
    print(red[1][1])
    print(red[1][2])
    print("blue")
    print(blue[0][0])
    print(blue[0][1])
    print(blue[0][2])
    print(blue[1][0])
    print(blue[1][1])
    print(blue[1][2])
    print("green")
    print(green[0][0])
    print(green[0][1])
    print(green[0][2])
    print(green[1][0])
    print(green[1][1])
    print(green[1][2])

def createRedOnly(animal_arr):
    for i in range(len(animal_arr)):
        for j in range(len(animal_arr[0])):
            for k in range(len(animal_arr[0][0])):
                if(k!=0):
                    animal_arr[i][j][k] = 0
    # print(animal_arr)
    return animal_arr

def createOneColor(color_one,idx):
    height_px = len(color_one)
    width_px = len(color_one[0])
    red = [[0 for c in range(width_px)] for r in range(height_px)]
    for i in range(height_px):
        for j in range(width_px):
            red[i][j] = color_one[i][j][idx]
    return red
    

    

dogs = loadDataSet(root, 'dogs')
cats = loadDataSet(root, 'cats')
createMatrix(createPath(root,'dogs'), dogs, 0)
