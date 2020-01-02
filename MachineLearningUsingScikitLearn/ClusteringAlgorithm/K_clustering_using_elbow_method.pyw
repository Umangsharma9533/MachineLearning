#Importing Relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

#Read the data

raw_data = pd.read_csv('Countries-exercise.csv')
raw_data

#Remove unwanted columms in this case it is name
data=raw_data.copy()
data=data.drop('name',axis=1)

#plot data for initial reference
plt.scatter(data['Longitude'],data['Latitude'])
plt.xlabel('')#specify the name of x-axis
plt.ylabel('')#specify the name of y-axis
plt.xlim(-200,200)#specify the limit of the graph
plt.ylim(-200,200)#specify the limit of the graph
plt.show()#show the above done changes

#calculate WCSS(Within clusters sum of squares
#List having values of wcss within clusters

wcss=[]
for i in range(1,11)#maximum clusters in current example is 11
    kmeans=KMeans(i)
	kmeans.fit_predict()
    wcss_inertia=kmeans.inertia_
	wcss.append(wcss_inertia)
wcss

#plot the graph between clusters and wcss and check the graph at what specific number of cluster the wcss and division of clusters will be optimal
number_of_clusters=range(1,11)
plt.plot(number_of_clusters,wcss)

#re-plot the graph based on the result obtained after the elbow method

kmeans=KMeans(3)
kmeans.fit(x)
result=kmeans.fit_predict(x)
data['Cluster']=result
plt.scatter(data['Latitude'],data['Longitude'],c=data['Cluster'],cmap='rainbow')
