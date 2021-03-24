########## USING K-MEANS CLUSTERING ##########

# implementing all the required libraries to perform clustering #
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

#importing dataset
crim = pd.read_csv("crime_data.csv")

# Implementing Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalizing the imported data frame (considering the numerical part of data)
df_norm = norm_func(crim.iloc[:,1:])
df_norm.head()

# drawing scree plot or elbow curve to obtain optimum K-value
k = list(range(2,10))

TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4).fit(df_norm)
model.labels_ # getting the labels of clusters assigned to each row 
ml=pd.Series(model.labels_)  # converting numpy array into pandas series object 

crim['clust']=ml # creating a  new column and assigning it as new column 
crim.clust.value_counts() #counting no. of data point under each cluster
crim = crim.iloc[:,[5,0,1,2,3,4]] #arranging the sequence of columns
crim.head()

# getting aggregate mean of each cluster
crime=crim.iloc[:,1:12].groupby(crim.clust).mean()

#creating a csv file and saving the result in .csv format
crim.to_csv("crime_result.csv")


########## USING HIERARCHICAL CLUSTERING ##########

# implementing all the required libraries to perform clustering #
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 

#importing dataset
crim = pd.read_csv("crime_data.csv")

# Implementing Normalization function 
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalizing the imported data frame (considering the numerical part of data)
df_norm = norm_func(crim.iloc[:,1:])
df_norm.describe()


# creating dendrogram to achieve optimum k-value
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# Applying AgglomerativeClustering choosing 4 as clusters from the dendrogram
from sklearn.cluster import	AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=4,linkage='complete',affinity = "euclidean").fit(df_norm) 
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
cluster_labels.value_counts()

crim['clust']=cluster_labels # creating a  new column and assigning it to new column 
crim = crim.iloc[:,[5,0,1,2,3,4]] #arranging the sequence of columns
crim.head()

# getting aggregate mean of each cluster
crime=crim.groupby(crim.clust).mean()

#creating a csv file and saving the result in .csv format
crim.to_csv("crime_result.csv",index=False) 
