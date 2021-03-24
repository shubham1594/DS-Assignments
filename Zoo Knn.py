# Importing Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
zoo = pd.read_csv("zoo.csv")

# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo,test_size = 0.2,random_state=0) # 0.2 => 20 percent of entire data 


## KNN using sklearn ##

# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

### Model Building ###

z_acc = [] # creating empty list variable 

# Building KNN model for 3 to 20 nearest neighbours(odd numbers)
for i in range(3,20,2):
    neigh = KNC(n_neighbors=i) 
    neigh.fit(train.iloc[:,1:17],train.iloc[:,17]) # fitting the model with training data
    train_z_acc = np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17]) #train accuracy 
    test_z_acc = np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17]) # test accuracy
    z_acc.append([train_z_acc,test_z_acc]) # storing the accuracy values in "z_acc" list


# Visualizations to identify most suitable model with better accuracy.
plt.plot(np.arange(3,20,2),[i[0] for i in z_acc],"bo-") # train accuracy plot 
plt.plot(np.arange(3,20,2),[i[1] for i in z_acc],"ro-") # test accuracy plot
plt.legend(["train","test"])

    
    
    
    
    