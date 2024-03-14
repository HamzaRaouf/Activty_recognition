# Reading Training Data from  files and save into appropriate formet
import numpy as np
from sklearn.svm import SVC  # for Model Selection
# for Test Score
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix

train_labels_for_openDoor = np.load('activity_monitoring/training/train_labels_OpenDoor_RubHands.npy')
train_labels_for_accelerometer = np.load('activity_monitoring/training/train_MSAccelerometer_OpenDoor_RubHands.npy')
trainGyroscopeData = np.load('activity_monitoring/training/train_MSGyroscope_OpenDoor_RubHands.npy')


print("Dimension of Label Data :",train_labels_for_openDoor.ndim)
print("Shape of Lebel Data  :",train_labels_for_openDoor.shape)

print("Dimension of train_labels_for_accelerometer Data :",train_labels_for_accelerometer.ndim)
print("Shape of train_labels_for_accelerometer Data :",train_labels_for_accelerometer.shape)

print("Dimension of trainGyroscopeData Data :",trainGyroscopeData.ndim)
print("Shape of trainGyroscopeData Data :",trainGyroscopeData.shape)

# As shape of both data is same so I'm Going to concatenate for X Training Data
x_train = train_labels_for_accelerometer+trainGyroscopeData
y_train = train_labels_for_openDoor


# Choose the SVM model
svm_model = SVC(kernel='linear')  
# Reshape the data into a 2D array becacuse SVC prcess with 2-D Data
X_flat = x_train.reshape(x_train.shape[0], -1)
print("===========>> Shape of Data NO is  :",X_flat.shape)

svm_model.fit(X_flat,y_train)


# Using evaluation methods like confusion matrix, accuracy, Average F1 score etc
test_X = np.load('activity_monitoring/testing/test_MSAccelerometer_OpenDoor_RubHands.npy') + np.load('activity_monitoring/testing/test_MSGyroscope_OpenDoor_RubHands.npy')
# convet 3d data into 2D Data
test_flat_X = test_X.reshape(test_X.shape[0],-1)
test_Y = np.load('activity_monitoring/testing/test_labels_OpenDoor_RubHands.npy')

print("-----Test X Shape is :",test_flat_X.shape)

y_predict = svm_model.predict(test_flat_X)
accurasy=accuracy_score(test_Y,y_pred=y_predict)
print("Accuracy Score is  :",accurasy)


# confusionMatrixScore=confusion_matrix(test_Y,y_predict)
# print("Confusion Matrix Score :",confusionMatrixScore)

