import pandas as pd
import numpy as np

Company = pd.read_csv("Company_Data.csv")
Company.head()
Company['Sales'].unique()
Company.Sales="category" # Converting "Sales" into categorical variable
type(Company.Sales)
Company.Sales.value_counts()
Company = Company.iloc[:,[0,1,2,3,4,5,7,8,6,9,10]] #arranging the sequence of columns
colnames = list(Company.columns)
predictors = colnames[1:8]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(Company,test_size = 0.2,random_state=0)

## 
from sklearn.tree import  DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
type(preds)
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

temp = pd.Series(model.predict(train[predictors])).reset_index(drop=True)

# Accuracy = train
np.mean(pd.Series(train.Sales).reset_index(drop=True) == pd.Series(model.predict(train[predictors])))
#1

# Accuracy = Test
np.mean(preds==test.Sales) # 1



