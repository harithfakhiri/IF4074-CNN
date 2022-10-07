from utils import Utils
from CNN import CNN
from ConvolutionalLayer import ConvolutionLayer
from DenseLayer import DenseLayer
from FlattenLayer import FlattenLayer

# def countAccuracy(image_name, label_array, label_trained):
#     score = 0
#     n = len(label_array)
#     for i in range(n):
#         if(label_array[i] == label_trained[i]):
#             score+=1
#         if(i<len(image_name[0])):
#             print(image_name[0][i], "--> label : ", label_name(label_array[i]), "and predicted as :", label_name(label_trained[i]))
#         else:
#             print(image_name[1][i-len(image_name[0])], ": label = ", label_name(label_array[i]), "and predicted as :", label_name(label_trained[i]))
#         print()
#         print("============================================================")
#     final = float(score/n)
#     return final

# def label_name(prediction):
#     if(prediction==0):
#         return "dogs"
#     elif(prediction==1):
#         return "cats"


utils = Utils()

#Load all image
matrix_image = utils.loadTrainDataSet()

preprocess_mat, label_array= utils.loadAllData(matrix_image)

# print(label_array)
dataset_num = len(preprocess_mat)
channel = len(preprocess_mat[0])
wh = len(preprocess_mat[0][0])

#inisiasi CNN
cnn = CNN(
    ConvolutionLayer(filter_size=8, num_filter=3,  num_channel=channel, 
                    isMax=True, act_func_detection="relu", stride=5, padding=0),
    ConvolutionLayer(filter_size=2, num_filter=2,  num_channel=channel, 
                    isMax=True, act_func_detection="relu", stride=1, padding=0),
    FlattenLayer(),
    DenseLayer(n_units=100, activation_function='relu'),
    DenseLayer(n_units=10, activation_function='relu'),
    DenseLayer(n_units=1, activation_function='sigmoid'),

)


result = cnn.predict(features=preprocess_mat,
    target=label_array,
    batch_size=10,
    epochs=7,
    learning_rate=0.05)


# acc_score = countAccuracy(matrix_image, label_array,result)

print()
# print("the final accuracy score is", acc_score)