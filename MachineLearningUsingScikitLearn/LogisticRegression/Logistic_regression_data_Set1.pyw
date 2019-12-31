#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()

#Load Data into data frame
raw_data=pd.read_csv('Bank-data.csv')
raw_data

#Copy data to a new variable to remove unwanted data and convert Value yes to 1 and no to 0
data=raw_data.copy()
data=data.drop('Unnamed: 0',axis=1)
data['y']=data['y'].map({'yes':1,'no':0})
data

#Declare dependent and independent variables
#Independent variable
x1=data['duration']
#dependent variable
y=data['y']

#Plotting regression
x=sm.add_constant(x1)
reg_log=sm.Logit(y,x)
results_summary=reg_log.fit()
#checking summary of regression
results_summary.summary()
#plot graph for provided data
plt.scatter(x1,y)

##### include other variables in independent variable to check what impact it will create on regression output
x1=data[['duration','interest_rate','march','credit','previous']]
y=data['y']
#Regression

x=sm.add_constant(x1)
reg_log=sm.Logit(y,x1)
results_summary=reg_log.fit()

results_summary.summary()

#checking the accuracy rate
def confusion_matrix(data,actual_values,model):
        
        # Confusion matrix 
        
        # Parameters
        # ----------
        # data: data frame or array
            # data is a data frame formatted in the same way as your input data (without the actual values)
            # e.g. const, var1, var2, etc. Order is very important!
        # actual_values: data frame or array
            # These are the actual values from the test_data
            # In the case of a logistic regression, it should be a single column with 0s and 1s
            
        # model: a LogitResults object
            # this is the variable where you have the fitted model 
            # e.g. results_log in this course
        # ----------
        
        #Predict the values using the Logit model
        pred_values = model.predict(data)
        # Specify the bins 
        bins=np.array([0,0.5,1])
        # Create a histogram, where if values are between 0 and 0.5 tell will be considered 0
        # if they are between 0.5 and 1, they will be considered 1
        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
        # Calculate the accuracy
        accuracy = (cm[0,0]+cm[1,1])/cm.sum()
        # Return the confusion matrix and 
        return cm, accuracy
        
        
confusion_matrix(x1,y,results_summary)
#Test the model

# Testing the model, load the data

test_data=pd.read_csv('Bank-data-testing.csv')

test_data
#Preprocess the data

test_data['y']=test_data['y'].map({'yes':1,'no':0})
test_data=test_data.drop('Unnamed: 0',axis=1)
test_data
#Declaring dependent and independent variable

x1_test=test_data[['duration','interest_rate','march','credit','previous']]
y_test=test_data['y']
x_test=sm.add_constant(x1_test)
reg_log=sm.Logit(y_test,x1_test)
results_summary=reg_log.fit()
#predict
confusion_matrix(x1_test,y_test,results_summary)
