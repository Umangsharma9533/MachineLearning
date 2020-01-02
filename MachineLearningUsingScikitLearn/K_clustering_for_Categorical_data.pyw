#importing relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set
from sklearn.cluster import KMeans

#Importing data
raw_data=pd.read_csv('Categorical.csv')
raw_data

#Preprocess the data
data=raw_data.copy()
data=data.drop('name',axis=1)
data

#OR#####
data=raw_data.copy()

#mapping the numerical data with numbers
data['continent'] = data['continent'].map({'North America':0,'Europe':1,'Asia':2,'Africa':3,'South America':4, 'Oceania':5,'Seven seas (open ocean)':6, 'Antarctica':7})
data

#iloc(rows_to_preserve, columns_to_preserve)
x=data.iloc[:,2:3]
x

#Creating a object for Kmeans() and forming 2 clusters
kmeans=KMeans(4)
prediction_output=kmeans.fit_predict(x)

#adding the prediction column in the data frame or table
data_copy['Cluster']=prediction_output

#Plotting a graph to see the output
plt.scatter(data_copy['Longitude'],data_copy['Latitude'], c=data_copy['Cluster'], cmap='rainbow')
plt.ylabel('Longitude',fontsize=20)
plt.xlabel('Latitude',fontsize=20)
