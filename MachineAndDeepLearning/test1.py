#Import relevant libraries
#Panda : Used for Data Manipulation
#Sklearn:Used for importing Logistic Regression model, split data into train and test data sets, Reshuffling of data,For feature selection
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_selection import RFE

#Loading the data in Dataframe
raw_data=pd.read_csv('HR_comma_sep.csv')
raw_data
#Creating a checkpoint by copying raw data into new data frame so that raw data will be preserved and make easy to backtrack if required
df=raw_data.copy()
#Make display limit of columns and rows to None 
pd.options.display.max_rows=None
pd.options.display.max_columns=None
display(df)

#Splitting data into two different DataFrame, required to balance the data
df_left_yes=df[df['left']==1]
df_left_no=df[df['left']==0]
#Picking random value from one DataFrame
df_left_no=resample(df_left_no,replace=True,n_samples=sum(df['left']),random_state=123)
#Concatinating bith subsets into one datasets
df_final=pd.concat([df_left_no,df_left_yes])
display(df_final)
#Reshuffle the data
df_ready_to_split=df.reindex(np.random.permutation(df_final.index))
display(df_ready_to_split)
df_ready_to_split.count()

#Creating dunmmies for categorical data
df_dept_dummies=pd.get_dummies(df_ready_to_split['Department'])
df_with_duumy_sal=pd.get_dummies(df_with_dummies['salary'])
#Concatinating dummies into the data frame
df_with_dummies=pd.concat([df_ready_to_split, df_dept_dummies], axis=1)
df_with_dummy_sal=pd.concat([df_with_dummies,df_with_duumy_sal],axis=1)
#Drop the categorical columns after creation and addition of dummies inot DataFrame to remove ambiquity
df_with_dummy_sal=df_with_dummy_sal.drop(['Department','salary'],axis=1)
#Setting the target and the  inputs for  the model
targets=df_with_dummy_sal['left']
inputs=df_with_dummy_sal.drop(['left'],axis=1)

#Standardizing the inputs
from sklearn.preprocessing import StandardScaler
ab_scaler=StandardScaler()
ab_scaler.fit(inputs)
inputs=ab_scaler.transform(inputs)

#Splitting data into train and test datasets
x_train,x_test,y_train,y_test=train_test_split(inputs,targets,test_size=0.2)
#Creating object Logistic Regression to feed the data
reg=LogisticRegression()
#Using of RFE for feature selection RFE(model,number_of_features_required)
regg=RFE(reg,10)
#Training the model
regg.fit(x_train,y_train)
#Checking score of training
regg.score(x_train,y_train)
#Checking accuracy of test data
regg.score(x_test,y_test)
