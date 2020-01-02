#importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set
from sklearn.cluster import KMeans

#Importing data
raw_data=pd.read_csv('Countries-exercise.csv')
raw_data

#Preprocess the data
data=raw_data.copy()
data=data.drop('name')
#OR#####
data=raw_data.copy()

#iloc(rows_to_preserve, columns_to_preserve)
data=data.iloc[:,1:3]

#Plotting data for initial look
plt.scatter(data_copy['Longitude'],data_copy['Latitude'])
plt.ylabel('Longitude',fontsize=20)
plt.xlabel('Latitude',fontsize=20)

#Creating a variable
x=data_copy['Longitude','Latitude']

#Creating a object for Kmeans() and forming 2 clusters
kmeans=KMeans(2)
prediction_output=kmeans.fit_predict(x)

#adding the prediction column in the data frame or table
data_copy['Cluster']=prediction

#Plotting a graph to see the output
plt.scatter(data_copy['Longitude'],data_copy['Latitude'], c=data_copy['Cluster'], cmap='rainbow')
plt.ylabel('Longitude',fontsize=20)
plt.xlabel('Latitude',fontsize=20)
