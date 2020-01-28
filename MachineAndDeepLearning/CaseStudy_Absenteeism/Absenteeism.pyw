#Import Relevant Libraries
import pandas as pd
#Load the data
raw_data=pd.read_csv("Absenteeism-data.csv")
#check data stored
raw_data

#Create a copy of the raw data (This is also called checkpoints). It make easy to rollback changes if required, so no need to run all the cells 
df=raw_data.copy()
#Make Maximum rows and columns to be visible to None , means we can display N number of rows and columns
pd.options.display.max_columns=None
pd.options.display.max_rows=None

#Check the info of the data frame to check for all the entries and its types
df.info()
pd.options.display.max_columns=None
pd.options.display.max_rows=None
#display() will display the full data frame
display(df)

#ID columns is dropped as it will not contribute in our Prediction /Machine Learning Algorithm
df=df.drop(['ID'],axis=1)
#We have number in column Reason for absence , in order to make it suitable for regresion we need to convert it into dummy variable
reason_for_abs=pd.get_dummies(df['Reason for Absence'],drop_first=True)

reason_for_abs
#Once dummies are created drop Reason for Absence column from main data frame as it can lead to duplivation(Multicollinearity)
df=df.drop(['Reason for Absence'],axis=1)

#It will move from left to right across row for each from 1:14 and so on
#Ecah reason time is broadly generalize into one reason
#For eg Reason1 is related to some sort of disease
#Reason2 related to giving birth or pregnancy
#Reason3 related to poisioning or not else where categorize
#Reason4 is related to minor health issues or checkup appointment
reason_type1=reason_for_abs.loc[:,1:14].max(axis=1)
reason_type2=reason_for_abs.loc[:,15:17].max(axis=1)
reason_type3=reason_for_abs.loc[:,18:21].max(axis=1)
reason_type4=reason_for_abs.loc[:,22:].max(axis=1)

#Adding the above created column in the dataframe
df=pd.concat([df,reason_type1,reason_type2,reason_type3,reason_type4],axis=1)
df.columns.values
#This will rename the existing columns
column_name=['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours','Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
df.columns=column_name
#Reordering the column in dataframe
column_name=['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']
df=df[column_name]
