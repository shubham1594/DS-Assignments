## Bagged Decision Trees for Classification ##

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#importing data set
fraud =pd.read_csv("Fraud_check.csv")
fraud.TI="category" # Converting "Taxable.Income" into categorical variable(Taxable.Income=TI
fraud = fraud.iloc[:,[2,3,4,0,1,5]] #arranging the sequence of columns

array_f = fraud.values
X = array_f[:,1:3]
Y = array_f[:,0]

seed = 7
num_trees = 100

kfold = KFold(n_splits=20, random_state=seed)
cart = DecisionTreeClassifier()
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())



## Random Forest Classification ##

from sklearn.ensemble import RandomForestClassifier

array_f = fraud.values
X = array_f[:,1:3]
Y = array_f[:,0]

num_trees = 50
max_features = 3

kfold = KFold(n_splits=10, random_state=7)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


## AdaBoost Classification ##

from sklearn.ensemble import AdaBoostClassifier

array_f = fraud.values
X = array_f[:,1:3]
Y = array_f[:,0]

num_trees = 10
seed=7

kfold = KFold(n_splits=30, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

