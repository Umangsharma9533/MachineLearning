import pandas as pd
raw_data=pd.read_csv("Absenteeism-data.csv")
raw_data

df=raw_data.copy()
pd.options.display.max_columns=None
pd.options.display.max_rows=None

df.info()
pd.options.display.max_columns=None
pd.options.display.max_rows=None
display(df)
df=df.drop(['ID'],axis=1)
reason_for_abs=pd.get_dummies(df['Reason for Absence'],drop_first=True)

reason_for_abs

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

df=pd.concat([df,reason_type1,reason_type2,reason_type3,reason_type4],axis=1)
df.columns.values
column_name=['Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours','Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
df.columns=column_name
column_name=['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education',
       'Children', 'Pets', 'Absenteeism Time in Hours']
df=df[column_name]
