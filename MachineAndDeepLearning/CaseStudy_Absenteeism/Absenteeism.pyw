import pandas as pd
raw_data=pd.read_csv("Absenteeism-data.csv")
raw_data

df=raw_data.copy()
pd.options.display.max_columns=None
pd.options.display.max_rows=None

df.info()