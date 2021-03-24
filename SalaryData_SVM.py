## importing Libraries
import pandas as pd
import numpy as np 

##Importing Dataset
SD_Train = pd.read_csv("SalaryData_Train(1).csv")
SD_Test = pd.read_csv("SalaryData_Test(1).csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

# Implementing train_test_split on dataset 
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    SD_Train[i] = number.fit_transform(SD_Train[i])
    SD_Test[i] = number.fit_transform(SD_Test[i])

train_X = SD_Train.iloc[:,0:13]
train_y = SD_Train.iloc[:,13]
test_X  = SD_Test.iloc[:,0:13]
test_y  = SD_Test.iloc[:,13]

## Create SVM classification object ##
from sklearn.svm import SVC

## IMPLEMENTING SVM model using 'sigmoid','poly' & 'rbf'  Kernel

# kernel = sigmoid
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_X,train_y)
pred_test_sigmoid = model_sigmoid.predict(test_X)
pd.crosstab(pred_test_sigmoid,test_y)
np.mean(pred_test_sigmoid==test_y) # Accuracy = 75.67

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
pd.crosstab(pred_test_poly,test_y)
np.mean(pred_test_poly==test_y) # Accuracy = 77.95

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
pd.crosstab(pred_test_rbf,test_y)
np.mean(pred_test_rbf==test_y) # Accuracy = 79.64
