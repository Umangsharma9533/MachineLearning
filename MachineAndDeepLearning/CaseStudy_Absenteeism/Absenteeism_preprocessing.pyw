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

#Create a checkpoints
df_reason_mod=df.copy()
df_reason_mod

#Converting string to time stamp and formatting it into date,month and Year format
df_reason_mod['Date']=pd.to_datetime(df_reason_mod['Date'],format='%d/%m/%Y')
df_reason_mod['Date']

#Create a empty list 
list_months=[]
list_months
#Check the shape of the data frame
df_reason_mod.shape
#Function to seperate month from the date
for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)
df_reason_mod['Month Value']=list_months
df_reason_mod

##Extract the day of the week
def day_of_the_week(date_value):
    return date_value.weekday()
#Apply will aplly the user defined function
df_reason_mod['Day Value']=df_reason_mod['Date'].apply(day_of_the_week)

#Now we have day and month so we will drop date column to prevent multicollinearity
df_cp=df_cp.drop(['Date'],axis=1)
df_cp.columns.values


#change the order of the columns in the data frame
update_columns=['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Month Value',
       'Day Value','Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']
df_cp=df_cp[update_columns]
df_reason_date_mod=df_cp

#This will check the counts /number of time the value occurs
df_reason_date_mod['Education'].value_counts()

## From the output above we have following findings
### Meaning of different number
#### 1: High School 
#### 2: Graduate
#### 3: Post Graduate
#### 4: Doctor or Master
### First for the most of the value we have high school so means we should not classify further into graduate, PG etc, 
######So we should map 0: to high school and 1: for graduation and above
df_reason_date_mod['Education']=df_reason_date_mod['Education'].map({1:0,2:1,3:1,4:1})

data_preprocessed=df_reason_date_mod.copy()

#Export the data to excel sheet
data_preprocessed.to_excel('output.xlsx')
