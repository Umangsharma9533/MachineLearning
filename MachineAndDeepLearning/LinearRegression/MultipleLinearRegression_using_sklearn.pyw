#....Problem Statement
#You are given a real estate dataset.

#Real estate is one of those examples that every regression course goes through as it is extremely easy to understand and there is a (almost always) certain causal relationship to be found.

#The data is located in the file: 'real_estate_price_size_year.csv'.

#You are expected to create a multiple linear regression (similar to the one in the lecture), using the new data.

#Apart from that, please:

#Display the intercept and coefficient(s)
#Find the R-squared and Adjusted R-squared
#Compare the R-squared and the Adjusted R-squared
#Compare the R-squared of this regression and the simple linear regression where only 'size' was used
#Using the model make a prediction about an apartment with size 750 sq.ft. from 2009
#Find the univariate (or multivariate if you wish - see the article) p-values of the two variables. What can you say about them?
#Create a summary table with your findings


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
x=data[['size','year']]
y=data['price']

#Regression 
reg=LinearRegression()
reg.fit(x,y)

#intercept calculation
reg.intercept_

#Coefficients calculation
reg.coef_

#R-squared Linear Value
r2=reg.score(x,y)
r2

#ADjusted R squared
n=x.shape[0]
p=x.shape[1]
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2

#predict the value
reg.predict([[750,2009]])

#Create summary of your Findings
from sklearn.feature_selection import f_regression
p_values=f_regression(x,y)[1]
reg_summary = pd.DataFrame(data = x.columns.values, columns=['Features'])
reg_summary ['Coefficients'] = reg.coef_
reg_summary ['p-values'] = p_values.round(3)
reg_summary