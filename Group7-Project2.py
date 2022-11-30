"""
COMP 309 - Data Warehousing & Predictive Analytics
Group Project 2
@author: Group 7
"""

"""
1. Data Exploration
"""

#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb 

#Load CSV file into dataframe
data_bicycle_thefts = pd.read_csv('C:\COMP309-Group7-Project2\Bicycle_Thefts.csv')

#Print first 10 rows of dataframe
print(data_bicycle_thefts.head(10))

#Print shape of dataframe: (number of rows, number of columns)
print(data_bicycle_thefts.shape)

#Print number of columns
number_columns = len(data_bicycle_thefts.columns)
print("Number of Columns: ", number_columns)

#Print number of rows
number_rows = len(data_bicycle_thefts)
print("Number of Rows: ", number_rows)

#Get array of column values of dataframe
column_values = data_bicycle_thefts.columns.values
for col in column_values:
    print(col)

#Generate desriptive statistics of dataset
data_bicycle_thefts.describe()

#Get mean, median, mode
df_mean = data_bicycle_thefts.mean()
df_median = data_bicycle_thefts.median()
df_mode = data_bicycle_thefts.mode()
print(df_mean)
print(df_median)
print(df_mode)

#Generate dataframe with correlation coefficients between columns
df_correlation = data_bicycle_thefts.corr()
print(df_correlation)

#Visualize correlation coefficients using the Seaborn library
sb.heatmap(df_correlation);

#Get info about each column in the dataframe
data_bicycle_thefts.info()

#Get Data Type for each column
print("Data Types: ", data_bicycle_thefts.dtypes)

#Get count of categories in the Status column
status_count = data_bicycle_thefts.Status.value_counts()
print(status_count)

#Get number of null values for each column
count_missing_values = data_bicycle_thefts.isna().sum()
print(data_bicycle_thefts.isna().sum())

#Get total number of null values in dataset
total_missing_values = data_bicycle_thefts.isna().sum().sum()
print("Total Number of missing values: ", total_missing_values)


#######################################
######### 2. Data Modelling ###########
#######################################


#Missing Data evaluations
#From the count_missing_values Series we notice that 3 columns include null values (Bike_Model, Bike_Colour, Cost_of_Bike)
#Use Imputation to fill the missing values

#Fill the missing values in the Bike_Model and Bike_Colour columns with "unknown"
data_bicycle_thefts["Bike_Model"].fillna("UNKNOWN", inplace = True) 
data_bicycle_thefts["Bike_Colour"].fillna("UNKNOWN", inplace = True)

#Fill the missing values in the Cost_of_Bike column with the average value in the column
print("Average Cost of Bike: ", int(data_bicycle_thefts['Cost_of_Bike'].mean()))
data_bicycle_thefts['Cost_of_Bike'].fillna(int(data_bicycle_thefts['Cost_of_Bike'].mean()),inplace=True)

#Check no null values are present in any column after Imputation
data_bicycle_thefts.isna().sum()

#Download dataframe as csv
data_bicycle_thefts.to_csv(r'C:\COMP309-Group7-Project2\Bicycle_Thefts_2.0.csv', index=False, header=True)

# Removing all rows where the status of the bike is UNKNOWN
# as it will not be useful in the predictive analysis
data_bicycle_thefts = data_bicycle_thefts[data_bicycle_thefts.Status != 'UNKNOWN']
# Removing all rows where the HoodID string "NSA"
data_bicycle_thefts = data_bicycle_thefts[data_bicycle_thefts.Hood_ID != 'NSA']

# as now only int values exit in HOOD_ID, setting it to int data type
data_bicycle_thefts['Hood_ID'] = pd.to_numeric(data_bicycle_thefts['Hood_ID'])

# Dropping some columns that are totally irrelvant
data_bicycle_thefts.drop('X', axis='columns', inplace=True)
data_bicycle_thefts.drop('Y', axis='columns', inplace=True)
data_bicycle_thefts.drop('OBJECTID', axis='columns', inplace=True)
data_bicycle_thefts.drop('event_unique_id', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_Date', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_Month', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_DayOfWeek', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_DayOfMonth', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_DayOfYear', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_Hour', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_Year', axis='columns', inplace=True)
data_bicycle_thefts.drop('City', axis='columns', inplace=True)
data_bicycle_thefts.drop('Longitude', axis='columns', inplace=True)
data_bicycle_thefts.drop('Latitude', axis='columns', inplace=True)
data_bicycle_thefts.drop('ObjectId2', axis='columns', inplace=True)

# Changing Status value to 0 or 1
data_bicycle_thefts['Status'] = [1 if status=='RECOVERED' else 0 for status in data_bicycle_thefts['Status']]

# Checking if any null values exist in the cleaned data
print("Total Number of missing values: ", data_bicycle_thefts.isna().sum().sum())

# Feature Selection: List of features that are most important for predictions as visualized from PowerBi report
features = ["Occurrence_DayOfWeek", "Occurrence_Hour", "Hood_ID", "Premises_Type", "Cost_of_Bike", "Status"]
featureSelection_df = data_bicycle_thefts[features]

# Store categorical values (column names) in an array
categorical_values = []
for col, col_type in featureSelection_df.dtypes.iteritems():
    if col_type == 'O':
        categorical_values.append(col)
print(categorical_values)

# Using get Dummies method to convert categorical (string) data to numerical
df_ohe = pd.get_dummies(featureSelection_df, columns=categorical_values, dummy_na=False)
print(df_ohe.head())
print(df_ohe.columns.values)
print(len(df_ohe) - df_ohe.count())

# Normalization/Standardization of the values with greater range to have same range
from sklearn import preprocessing
# Get column names first
names = df_ohe.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_ohe)
scaled_df = pd.DataFrame(scaled_df, columns=names)
print(scaled_df.head())
print(scaled_df.dtypes)

###################
#### BALANCE the IMBALANCED data here or after splitting the data????????????????????
###############

# Split the data into train test
x = scaled_df[scaled_df.columns.difference(['Status'])]
y = scaled_df['Status']
y = y.astype(int)

from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x,y, test_size = 0.2)

####################################################
########### 3. Predictive model building ###########
####################################################

# building logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(x, y)

# Performing 10 fold cross validation to score the model (test accuracy)
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)

testY_predict = lr.predict(testX)
testY_predict.dtype
#print(testY_predict)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=[0,9]))















