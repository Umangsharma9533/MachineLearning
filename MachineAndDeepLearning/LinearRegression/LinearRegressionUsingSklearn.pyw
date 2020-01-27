# importing all relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression

#Loading data into variablre using Pandas
data=pd.read_csv('real_estate_price_size.csv')
data.head()

#Declare the dependent and the independent variables
x=data['size']
y=data['price']

#plotting initial graph
plt.scatter(x,y)
plt.xlabel('Size',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.show()

#reshaping independent variable into 2d array as sklearn function accepts only nd arrays
x_matrix=x.values.reshape(-1,1)
x_matrix

#creating a object of linear regression class
reg=LinearRegression()
reg.fit(x_matrix,y)

#Plotting a graph with regression line

plt.scatter(x_matrix,y)
yhat=reg.coef_*x_matrix+reg.intercept_
fig=plt.plot(x_matrix,yhat,lw=2)
plt.xlabel('Size of property')
plt.ylabel('price of property')
plt.show()

#making prediction
new_Data=pd.DataFrame(data=[750],columns=['Price'])
reg.predict(new_Data)
