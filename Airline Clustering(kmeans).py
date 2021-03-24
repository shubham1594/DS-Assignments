########## USING K-MEANS CLUSTERING ##########

# implementing libraries #
import pandas as pd
import matplotlib.pylab as plt
from	sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
import numpy as np

#importing dataset
ew_Air = pd.read_excel("EW_Airlines.xlsx")

# Implementing Normalization function 
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalizing the imported data frame (considering the numerical part of data)
df_norm = norm_func(ew_Air.iloc[:,1:])
df_norm.head()

# drawing scree plot or elbow curve to obtain optimum K-value
k = list(range(2,15))

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

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5).fit(df_norm)
model.labels_ # getting the labels of clusters assigned to each row 
ml=pd.Series(model.labels_)  # converting numpy array into pandas series object 

ew_Air['clust']=ml # creating a new column and assigning it as new column 
ew_Air.clust.value_counts() #counting no. of data point under each cluster
ew_Air = ew_Air.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]] #arranging the sequence of columns
ew_Air.head()

# getting aggregate mean of each cluster
ewA=ew_Air.iloc[:,1:12].groupby(ew_Air.clust).mean()

#creating a csv file and saving the result in .csv format
ew_Air.to_csv("EWA_result.csv")  # saving the result in .csv format


########## USING HIERARCHICAL CLUSTERING ##########

# implementing libraries
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch

#importing dataset
ew_Air = pd.read_excel("EW_Airlines.xlsx")

# Implementing Normalization function 
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ew_Air.iloc[:,1:])
df_norm.describe()


# creating dendrogram to achieve optimum k-value
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(300, 100));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Applying Agglomerative Clustering choosing 4 as clusters from the dendrogram
from sklearn.cluster import	AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=5,linkage='complete',affinity = "euclidean").fit(df_norm) 
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
cluster_labels.value_counts()

ew_Air['clust']=cluster_labels # creating a  new column and assigning it as new column 
ew_Air = ew_Air.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]] #arranging the sequence of columns
ew_Air.head()

# getting aggregate mean of each cluster
ewA=ew_Air.iloc[:,2:].groupby(ew_Air.clust).mean()

#creating a csv file and saving the result in .csv format
ew_Air.to_csv("EWA_result.csv",index=False) 