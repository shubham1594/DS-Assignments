#importing libraries
import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt

#Importing Data
bank = pd.read_csv("bank-full.csv",sep=';')
#removing unneccessary variablles
bank.drop(["job","marital","education","default","housing","loan","contact","day","month","poutcome"],axis=1,inplace=True)
bank['y'] = bank['y'].map({'yes': 1, 'no': 0}) 
bank.head(4)


# checking for the missing/null values
bank.isnull().sum() 
# No null values found


import seaborn as sns
sns.boxplot(x="campaign",y="duration",data=bank)
plt.boxplot(bank.balance)
bank.describe()

# usage lambda and apply function
bank.apply(lambda x:x.mean()) 
bank.mean()

bank.y.value_counts()  # to find how many time a certain value has repeated
bank.y.value_counts().index[0] #to find the most occuring value
bank.y.mode()[0]

#Model building 
import statsmodels.formula.api as sm
logit_model = sm.logit('y~age+balance+campaign+pdays+previous',data = bank).fit()

#summary
logit_model.summary()
y_pred = logit_model.predict(bank)


bank["pred_prob"] = y_pred # Creating new column for storing predicted class of y

# filling all the cells with zeroes
bank["y_val"] = 0

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
bank.loc[y_pred>=0.5,"y_val"] = 1
bank.y_val

from sklearn.metrics import classification_report
classification_report(bank.y_val,bank.y)

# confusion matrix 
confusion_matrix = pd.crosstab(bank['y'],bank.y_val)
confusion_matrix
accuracy = (39885+10)/(45211) # 88.24

# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(bank.y, y_pred)


# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 


### Dividing data into train and test data sets
bank.drop("y_val",axis=1,inplace=True)

from sklearn.model_selection import train_test_split
train,test = train_test_split(bank,test_size=0.3)

# checking na values 
train.isnull().sum();test.isnull().sum()

# Building a model on train data set 
train_model = sm.logit('y~balance+campaign+pdays+previous',data = train).fit()

#summary
train_model.summary()
train_pred = train_model.predict(train.iloc[:,1:])

# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
train["train_pred"] = np.zeros(31647)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(train['y'],train.train_pred)
confusion_matrix
accuracy_train = (27928+8)/(31647) # 88.27

# Prediction on Test data set
test_pred = train_model.predict(test)

# Creating new column for storing predicted class of y
# filling all the cells with zeroes
test["test_pred"] = np.zeros(13564)

# taking threshold value as 0.5 and above the prob value will be treated as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test['y'],test.test_pred)
confusion_matrix
# accuracy test
accuracy_test = (11942+4)/(13564) # 88.07
