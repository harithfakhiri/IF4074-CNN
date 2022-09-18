from utils import Utils

utils = Utils()

matrix_image = utils.loadTrainDataSet()
print(matrix_image)

preprocess_mat, label_array= utils.loadAllData(matrix_image)
# print(preprocess_mat)
print("============")
print(label_array)

