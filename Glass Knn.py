# Importing Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
glass = pd.read_csv("glass.csv")

# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(glass,test_size = 0.2,random_state=0) # 0.2 => 20 percent of entire data 


## KNN using sklearn ##

# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

### Model Building ###

g_acc = [] # creating empty list variable 

# Building KNN model for 3 to 18 nearest neighbours(odd numbers)
for i in range(3,18,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9]) # fitting the model with training data
    train_g_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9]) #train accuracy
    test_g_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9]) # test accuracy
    g_acc.append([train_g_acc,test_g_acc]) # storing the accuracy values in "z_acc" list


# Visualizations to identify most suitable model with better accuracy.
plt.plot(np.arange(3,18,2),[i[0] for i in g_acc],"bo-") # train accuracy plot 
plt.plot(np.arange(3,18,2),[i[1] for i in g_acc],"ro-") # test accuracy plot
plt.legend(["train","test"])

    
    
    
    
    