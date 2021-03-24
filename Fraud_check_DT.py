import pandas as pd
import numpy as np

Fraud= pd.read_csv("Fraud_check.csv")
Fraud.head()
Fraud['TI'].unique() # Taxable.Income= TI
type(Fraud.TI)
Fraud.TI.value_counts()
Fraud.TI="category"
Fraud = Fraud.iloc[:,[2,3,4,0,1,5]] #arranging the sequence of columns
colnames = list(Fraud.columns)
predictors = colnames[1:3]
target = colnames[0]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test = train_test_split(Fraud,test_size = 0.2,random_state=0)

## Building Decision Tree Model
from sklearn.tree import  DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train[predictors],train[target])
preds = model.predict(test[predictors])
type(preds)
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)

temp = pd.Series(model.predict(train[predictors])).reset_index(drop=True)

# Accuracy = train
np.mean(pd.Series(train.TI).reset_index(drop=True) == pd.Series(model.predict(train[predictors])))#1

# Accuracy = Test
np.mean(preds==test.TI) # 1


