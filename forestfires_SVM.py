## importing Libraries
import pandas as pd 
import numpy as np 

##Importing Dataset
f_fires = pd.read_csv("forestfires.csv")
f_fires.columns

# Implementing train_test_split on dataset 
from sklearn.model_selection import train_test_split
train,test = train_test_split(f_fires,test_size = 0.3,random_state=0)
train_X = train.iloc[:,2:30]
train_y = train.iloc[:,30]
test_X  = test.iloc[:,2:30]
test_y  = test.iloc[:,30]

## Create SVM classification object ##
from sklearn.svm import SVC

## IMPLEMENTING SVM model using 'linear','poly' &'rbf' Kernel

# kernel = linear
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
pd.crosstab(pred_test_linear,test_y)
np.mean(pred_test_linear==test_y) # Accuracy = 98.71

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
pd.crosstab(pred_test_poly,test_y)
np.mean(pred_test_poly==test_y) # Accuracy = 75.64

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
pd.crosstab(pred_test_rbf,test_y)
np.mean(pred_test_rbf==test_y) # Accuracy = 72.43
