#importing libraries
import pandas as pd
import numpy as np

#imorting dataset
book_df = pd.read_csv('book.csv',engine = 'python')

#number of unique users in the dataset
len(book_df.userId.unique())
len(book_df.book.unique())

user_book_df = book_df.pivot_table(index='userId',
                                 columns='book',
                                 values='rating').reset_index(drop=True)

user_book_df
user_book_df.index = book_df.userId.unique()
user_book_df

#Impute those NaNs with 0 values
user_book_df.fillna(0, inplace=True)
user_book_df

#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances

user_sim = 1 - pairwise_distances( user_book_df.values,metric='cosine')
user_sim_df = pd.DataFrame(user_sim) #Storing the results in a dataframe

#Set the index and column names to user ids 
user_sim_df.index =book_df.userId.unique()
user_sim_df.columns = book_df.userId.unique()

user_sim_df.iloc[0:5, 0:5]
np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]

#Most Similar Users
user_sim_df.idxmax(axis=1)[0:5]

book_df[(book_df['userId']==276729) | (book_df['userId']==276726)]

user_1=book_df[book_df['userId']==276729]
user_2=book_df[book_df['userId']==276744]

user_2.book
user_1.book

book_merge=pd.merge(user_1,user_2,on='book',how='outer')

