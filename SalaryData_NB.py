#IMPORTING LIBRARIES
import pandas as pd
from sklearn.metrics import confusion_matrix

# IMPORTING DATASET
salary_train = pd.read_csv("SalaryData_Train.csv")
salary_test = pd.read_csv("SalaryData_Test.csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

## Implementing test_train_split on dataset
from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

# Splitting data into train and test
colnames = salary_train.columns
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]

# Building and predicting at the same time using GaussianNB
from sklearn.naive_bayes import GaussianNB
sgnb = GaussianNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
cm_g=confusion_matrix(testY,spred_gnb) # Confusion matrix GaussianNB model
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) #79.46

# Building and predicting at the same time using MultinomialNB
from sklearn.naive_bayes import MultinomialNB
smnb = MultinomialNB()
spred_mnb = smnb.fit(trainX,trainY).predict(testX)
cm_m=confusion_matrix(testY,spred_mnb)  # Confusion matrix multinomial model
print("Accuracy",(10891+780)/(10891+780+2920+780))  # 75.92%