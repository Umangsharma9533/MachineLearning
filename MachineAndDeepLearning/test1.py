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


























import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_selection import RFE
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

df_dept_dummies=pd.get_dummies(df_ready_to_split['Department'])


df_with_duumy_sal=pd.get_dummies(df_with_dummies['salary'])
df_with_dummy_sal=pd.concat([df_with_dummies,df_with_duumy_sal],axis=1)
df_with_dummy_sal=df_with_dummy_sal.drop(['Department','salary'],axis=1)
targets=df_with_dummy_sal['left']

inputs=df_with_dummy_sal.drop(['left'],axis=1)
from sklearn.preprocessing import StandardScaler
ab_scaler=StandardScaler()
ab_scaler.fit(inputs)
inputs=ab_scaler.transform(inputs)
x_train,x_test,y_train,y_test=train_test_split(inputs,targets,test_size=0.2)
reg=LogisticRegression()
regg=RFE(reg,10)
regg.fit(x_train,y_train)
regg.score(x_train,y_train)
regg.score(x_test,y_test)
df_with_dummies=pd.concat([df_ready_to_split, df_dept_dummies], axis=1)
