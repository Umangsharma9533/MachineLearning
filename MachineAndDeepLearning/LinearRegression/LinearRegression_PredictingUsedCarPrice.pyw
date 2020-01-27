#importing all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
#for checking multicollineriaty
from statsmodels.stats.outliers_influence import variance_inflation_factor

#reading data from csv file
raw_data=pd.read_csv('1.04. Real-life example.csv')
raw_data.head()

#data Preprocessing Starts from here

raw_data.describe(include='all')



#check for the null values and take sum of null values in total to check how many entries have null values
raw_data.isnull().sum()
#after checking null values its time to remove null values from the table use dropna() function for that
data_no_mv=raw_data.dropna(axis=0)
#Now check the table whether all the null values are removed from the calculation
data_no_mv.describe(include='all')

from sklearn.feature_selection import f_regression
x=data_no_mv[['Mileage','EngineV','Year']]
y=data_no_mv['Price']
f_regression(x,y)

#plot a graph of Price column of the table to chekc that values are normally distributed
sns.distplot(data_no_mv['Price'])


#If values are not normally distributed then remove top 1% of the value.
#Set reference to remove 1% of top outliers as we have lot of difference between mean and max values
q=data_no_mv['Price'].quantile(0.99)
data_1_price=data_no_mv[data_no_mv['Price']<q]
sns.distplot(data_1_price['Price'])


#check similar for other variable that if they are normally distributed or not
sns.distplot(data_1_price['EngineV'])

#After plotting the graph we check that this columns contains some invalid values so we are taking values which are valid 
data_2_EngineV=data_1_price[data_1_price['EngineV']<6.5]


#Now plot a graph to check whether it is normally distributed or not
sns.distplot(data_2_EngineV['EngineV'])


#Now plot a graph to check whether it is normally distributed or not
sns.distplot(data_2_EngineV['Mileage'])



#Remove the outliers from the year columns here we will preserve 1% of top values
#Now plot a graph to check whether it is normally distributed or not
q=data_2_EngineV['Year'].quantile(0.99)
data_3_Year=data_2_EngineV[data_2_EngineV['Year']<q]
sns.distplot(data_3_Year['Year'])


data_3_Year.describe(include ='all')


#Reset the index for the values in the table as in pandas index in the table is preserved so we need to reset those before processeding further
data_Cleaned=data_3_Year.reset_index(drop=True)


sns.pairplot(data_Cleaned)

# From the subplots and the PDF of price, we can easily determine that 'Price' is exponentially distributed
# A good transformation in that case is a log transformation
#So we need to log the values of price to remove exopential characteristics
log_price = np.log(data_Cleaned['Price'])

# Then we add it to our data frame
data_Cleaned['log_price'] = log_price
data_Cleaned
data_Cleaned=data_Cleaned.drop(['Price'],axis=1)

sns.pairplot(data_Cleaned)


# To make this as easy as possible to use, we declare a variable where we put
# all features where we want to check for multicollinearity
# since our categorical data is not yet preprocessed, we will only take the numerical ones
variables = data_Cleaned[['Mileage','Year','EngineV']]

# we create a new data frame which will include all the VIFs
# note that each variable has its own variance inflation factor as this measure is variable specific (not model specific)
vif = pd.DataFrame()

# here we make use of the variance_inflation_factor, which will basically output the respective VIFs 
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
# Finally, I like to include names so it is easier to explore the result
vif["Features"] = variables.columns

vif


# Since Year has the highest VIF, I will remove it from the model
# This will drive the VIF of other variables down!!! 
# So even if EngineV seems with a high VIF, too, once 'Year' is gone that will no longer be the case
data_no_multicollinearity = data_Cleaned.drop(['Year'],axis=1)

data_with_dummies=pd.get_dummies(data_no_multicollinearity,drop_first=True)

data_with_dummies.columns.values


cols=['log_price','Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes',
       'Model_100', 'Model_11', 'Model_116', 'Model_118', 'Model_120',
       'Model_19', 'Model_190', 'Model_200', 'Model_21', 'Model_210',
       'Model_220', 'Model_230', 'Model_25', 'Model_250', 'Model_300',
       'Model_316', 'Model_318', 'Model_320', 'Model_323', 'Model_324',
       'Model_325', 'Model_328', 'Model_330', 'Model_335', 'Model_428',
       'Model_4Runner', 'Model_5', 'Model_5 Series', 'Model_5 Series GT',
       'Model_520', 'Model_523', 'Model_524', 'Model_525', 'Model_528',
       'Model_530', 'Model_535', 'Model_540', 'Model_545', 'Model_550',
       'Model_6 Series Gran Coupe', 'Model_630', 'Model_640', 'Model_645',
       'Model_650', 'Model_730', 'Model_735', 'Model_740', 'Model_745',
       'Model_750', 'Model_760', 'Model_80', 'Model_9', 'Model_90',
       'Model_A 140', 'Model_A 150', 'Model_A 170', 'Model_A 180',
       'Model_A1', 'Model_A3', 'Model_A4', 'Model_A4 Allroad', 'Model_A5',
       'Model_A6', 'Model_A6 Allroad', 'Model_A7', 'Model_A8',
       'Model_ASX', 'Model_Amarok', 'Model_Auris', 'Model_Avalon',
       'Model_Avensis', 'Model_Aygo', 'Model_B 170', 'Model_B 180',
       'Model_B 200', 'Model_Beetle', 'Model_Bora', 'Model_C-Class',
       'Model_CL 180', 'Model_CL 500', 'Model_CL 55 AMG', 'Model_CL 550',
       'Model_CL 63 AMG', 'Model_CLA 200', 'Model_CLA 220',
       'Model_CLA-Class', 'Model_CLC 180', 'Model_CLC 200',
       'Model_CLK 200', 'Model_CLK 220', 'Model_CLK 230', 'Model_CLK 240',
       'Model_CLK 280', 'Model_CLK 320', 'Model_CLK 430', 'Model_CLS 350',
       'Model_CLS 500', 'Model_CLS 63 AMG', 'Model_Caddy', 'Model_Camry',
       'Model_Caravelle', 'Model_Carina', 'Model_Carisma', 'Model_Celica',
       'Model_Clio', 'Model_Colt', 'Model_Corolla', 'Model_Corolla Verso',
       'Model_Cross Touran', 'Model_Duster', 'Model_E-Class',
       'Model_Eclipse', 'Model_Eos', 'Model_Espace', 'Model_FJ Cruiser',
       'Model_Fluence', 'Model_Fortuner', 'Model_G 320', 'Model_G 350',
       'Model_G 500', 'Model_G 55 AMG', 'Model_G 63 AMG', 'Model_GL 320',
       'Model_GL 350', 'Model_GL 420', 'Model_GL 450', 'Model_GL 500',
       'Model_GL 550', 'Model_GLK 220', 'Model_GLK 300', 'Model_Galant',
       'Model_Golf GTI', 'Model_Golf II', 'Model_Golf III',
       'Model_Golf IV', 'Model_Golf Plus', 'Model_Golf V',
       'Model_Golf VI', 'Model_Golf VII', 'Model_Golf Variant',
       'Model_Grand Scenic', 'Model_Grandis', 'Model_Hiace',
       'Model_Highlander', 'Model_Hilux', 'Model_I3', 'Model_IQ',
       'Model_Jetta', 'Model_Kangoo', 'Model_Koleos', 'Model_L 200',
       'Model_LT', 'Model_Laguna', 'Model_Lancer',
       'Model_Lancer Evolution', 'Model_Lancer X',
       'Model_Lancer X Sportback', 'Model_Land Cruiser 100',
       'Model_Land Cruiser 105', 'Model_Land Cruiser 200',
       'Model_Land Cruiser 76', 'Model_Land Cruiser 80',
       'Model_Land Cruiser Prado', 'Model_Latitude', 'Model_Lite Ace',
       'Model_Logan', 'Model_Lupo', 'Model_M5', 'Model_M6', 'Model_MB',
       'Model_ML 250', 'Model_ML 270', 'Model_ML 280', 'Model_ML 320',
       'Model_ML 350', 'Model_ML 400', 'Model_ML 430', 'Model_ML 500',
       'Model_ML 550', 'Model_ML 63 AMG', 'Model_Mark II', 'Model_Master',
       'Model_Matrix', 'Model_Megane', 'Model_Modus', 'Model_Multivan',
       'Model_New Beetle', 'Model_Outlander', 'Model_Outlander XL',
       'Model_Pajero', 'Model_Pajero Pinin', 'Model_Pajero Sport',
       'Model_Pajero Wagon', 'Model_Passat B2', 'Model_Passat B3',
       'Model_Passat B4', 'Model_Passat B5', 'Model_Passat B6',
       'Model_Passat B7', 'Model_Passat B8', 'Model_Passat CC',
       'Model_Phaeton', 'Model_Pointer', 'Model_Polo', 'Model_Previa',
       'Model_Prius', 'Model_Q3', 'Model_Q5', 'Model_Q7', 'Model_R 320',
       'Model_R8', 'Model_Rav 4', 'Model_S 140', 'Model_S 250',
       'Model_S 280', 'Model_S 300', 'Model_S 320', 'Model_S 350',
       'Model_S 400', 'Model_S 420', 'Model_S 430', 'Model_S 500',
       'Model_S 550', 'Model_S 600', 'Model_S 63 AMG', 'Model_S 65 AMG',
       'Model_S4', 'Model_S5', 'Model_S8', 'Model_SL 500 (550)',
       'Model_SL 55 AMG', 'Model_SLK 200', 'Model_SLK 350',
       'Model_Sandero', 'Model_Scenic', 'Model_Scion', 'Model_Scirocco',
       'Model_Sequoia', 'Model_Sharan', 'Model_Sienna', 'Model_Smart',
       'Model_Space Star', 'Model_Space Wagon', 'Model_Sprinter',
       'Model_Sprinter 208', 'Model_Sprinter 210', 'Model_Sprinter 211',
       'Model_Sprinter 212', 'Model_Sprinter 213', 'Model_Sprinter 311',
       'Model_Sprinter 312', 'Model_Sprinter 313', 'Model_Sprinter 315',
       'Model_Sprinter 316', 'Model_Sprinter 318', 'Model_Sprinter 319',
       'Model_Symbol', 'Model_Syncro', 'Model_T2 (Transporter)',
       'Model_T3 (Transporter)', 'Model_T4 (Transporter)',
       'Model_T4 (Transporter) ', 'Model_T5 (Transporter)',
       'Model_T5 (Transporter) ', 'Model_T6 (Transporter)',
       'Model_T6 (Transporter) ', 'Model_TT', 'Model_Tacoma',
       'Model_Tiguan', 'Model_Touareg', 'Model_Touran', 'Model_Trafic',
       'Model_Tundra', 'Model_Up', 'Model_V 250', 'Model_Vaneo',
       'Model_Vento', 'Model_Venza', 'Model_Viano', 'Model_Virage',
       'Model_Vista', 'Model_Vito', 'Model_X1', 'Model_X3', 'Model_X5',
       'Model_X5 M', 'Model_X6', 'Model_X6 M', 'Model_Yaris', 'Model_Z3',
       'Model_Z4']
        
        
# To implement the reordering, we will create a new df, which is equal to the old one but with the new order of features
data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()





###linear 




targets = data_preprocessed['log_price']

# The inputs are everything BUT the dependent variable, so we can simply drop it
inputs = data_preprocessed.drop(['log_price'],axis=1)
# Import the scaling module
from sklearn.preprocessing import StandardScaler

# Create a scaler object
scaler = StandardScaler()
# Fit the inputs (calculate the mean and standard deviation feature-wise)
scaler.fit(inputs)

inputs_scaled = scaler.transform(inputs)

# Import the module for the split
from sklearn.model_selection import train_test_split

# Split the variables with an 80-20 split and some random state
# To have the same split as mine, use random_state = 365
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)

# Create a linear regression object
reg = LinearRegression()
# Fit the regression with the scaled TRAIN inputs and targets
reg.fit(x_train,y_train)

# Let's check the outputs of the regression
# I'll store them in y_hat as this is the 'theoretical' name of the predictions
y_hat = reg.predict(x_train)


# The simplest way to compare the targets (y_train) and the predictions (y_hat) is to plot them on a scatter plot
# The closer the points to the 45-degree line, the better the prediction
plt.scatter(y_train, y_hat)
# Let's also name the axes
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
# Sometimes the plot will have different scales of the x-axis and the y-axis
# This is an issue as we won't be able to interpret the '45-degree line'
# We want the x-axis and the y-axis to be the same
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# Another useful check of our model is a residual plot
# We can plot the PDF of the residuals and check for anomalies
sns.distplot(y_train - y_hat)

# Include a title
plt.title("Residuals PDF", size=18)

# In the best case scenario this plot should be normally distributed
# In our case we notice that there are many negative residuals (far away from the mean)
# Given the definition of the residuals (y_train - y_hat), negative values imply
# that y_hat (predictions) are much higher than y_train (the targets)
# This is food for thought to improve our model


# Find the R-squared of the model
reg.score(x_train,y_train)

# Note that this is NOT the adjusted R-squared
# in other words... find the Adjusted R-squared to have the appropriate measure :)


# Obtain the bias (intercept) of the regression
reg.intercept_



# Obtain the weights (coefficients) of the regression
reg.coef_

# Note that they are barely interpretable if at all


# Create a regression summary where we can compare them with one-another
reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary

# Check the different categories in the 'Brand' variable
data_Cleaned['Brand'].unique()

# In this way we can see which 'Brand' is actually the benchmark

# Once we have trained and fine-tuned our model, we can proceed to testing it
# Testing is done on a dataset that the algorithm has never seen
# Luckily we have prepared such a dataset
# Our test inputs are 'x_test', while the outputs: 'y_test' 
# We SHOULD NOT TRAIN THE MODEL ON THEM, we just feed them and find the predictions
# If the predictions are far off, we will know that our model overfitted
y_hat_test = reg.predict(x_test)


# Create a scatter plot with the test targets and the test predictions
# You can include the argument 'alpha' which will introduce opacity to the graph
plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()




df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()

# We can also include the test targets in that data frame (so we can manually compare them)
df_pf['Target'] = np.exp(y_test)
df_pf



# After displaying y_test, we find what the issue is
# The old indexes are preserved (recall earlier in that code we made a note on that)
# The code was: data_cleaned = data_4.reset_index(drop=True)

# Therefore, to get a proper result, we must reset the index and drop the old indexing
y_test = y_test.reset_index(drop=True)

# Check the result
y_test.head()


# Let's overwrite the 'Target' column with the appropriate values
# Again, we need the exponential of the test log price
df_pf['Target'] = np.exp(y_test)
df_pf



# Additionally, we can calculate the difference between the targets and the predictions
# Note that this is actually the residual (we already plotted the residuals)
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

# Since OLS is basically an algorithm which minimizes the total sum of squared errors (residuals),
# this comparison makes a lot of sense

# Finally, it makes sense to see how far off we are from the result percentage-wise
# Here, we take the absolute difference in %, so we can easily order the data frame
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf
# Finally, it makes sense to see how far off we are from the result percentage-wise
# Here, we take the absolute difference in %, so we can easily order the data frame
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf





# Sometimes it is useful to check these outputs manually
# To see all rows, we use the relevant pandas syntax
pd.options.display.max_rows = 999
# Moreover, to make the dataset clear, we can display the result with only 2 digits after the dot 
pd.set_option('display.float_format', lambda x: '%.2f' % x)
# Finally, we sort by difference in % and manually check the model
df_pf.sort_values(by=['Difference%'])

# Obtain the weights (coefficients) of the regression
reg.coef_
