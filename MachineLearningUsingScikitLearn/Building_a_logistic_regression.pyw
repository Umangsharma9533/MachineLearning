#importing relevant libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

#Loading data 
raw_data=pd.read_csv('Example-bank-data.csv')
raw_data

#preprocess data

data=raw_data.copy()
data.drop(['Unnamed: 0'],axis=1)
data['y']=data['y'].map({'yes':1,'no':0})
data

#Dependent and Independent variables

x1=data['duration']
y=data['y']
x=sm.add_constant(x1)
reg_log=sm.Logit(y,x)
results_log_summary=reg_log.fit()

#plot data

plt.scatter(x1,y)
plt.xlabel('Duration',fontsize=20)
plt.ylabel('Subscription',fontsize=20)