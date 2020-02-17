import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

raw_data=pd.read_csv('HR_comma_sep.csv')
raw_data
df=raw_data.copy()
pd.options.display.max_rows=None
pd.options.display.max_columns=None
display(df)

df_left_yes=df[df['left']==1]
df_left_no=df[df['left']==0]
df_left_no=resample(df_left_no,replace=True,n_samples=sum(df['left']),random_state=123)
df_final=pd.concat([df_left_no,df_left_yes])
display(df_final)

df_ready_to_split=df.reindex(np.random.permutation(df_final.index))
display(df_ready_to_split)
df_ready_to_split.count()


targets=df_ready_to_split['left']
inputs=df_ready_to_split.drop(['left','Department','salary'],axis=1)


targets.count()

inputs.count()


x_train,x_test,y_train,y_test=train_test_split(inputs,targets,test_size=0.2,random_state=20)


from sklearn.linear_model import LogisticRegression

reg=LogisticRegression()

x_train.shape
y_train.shape
reg.fit(x_train,y_train)

reg.score(x_train,y_train)