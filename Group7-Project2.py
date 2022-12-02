"""
COMP 309 - Data Warehousing & Predictive Analytics
Group Project 2
@author: Group 7
"""

"""
1. Data Exploration
"""
#######################################
######### 1. Data Exploration #########
#######################################

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


"""
2. Data Modelling
"""
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

#Download dataframe as csv for PowerBi
#data_bicycle_thefts.to_csv(r'C:\COMP309-Group7-Project2\Bicycle_Thefts_2.0.csv', index=False, header=True)

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
data_bicycle_thefts.drop('Report_Month', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_DayOfWeek', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_DayOfMonth', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_DayOfYear', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_Hour', axis='columns', inplace=True)
data_bicycle_thefts.drop('Report_Year', axis='columns', inplace=True)
data_bicycle_thefts.drop('Occurrence_DayOfMonth', axis='columns', inplace=True)
data_bicycle_thefts.drop('Occurrence_DayOfYear', axis='columns', inplace=True)
data_bicycle_thefts.drop('Occurrence_Year', axis='columns', inplace=True)
data_bicycle_thefts.drop('City', axis='columns', inplace=True)
data_bicycle_thefts.drop('Longitude', axis='columns', inplace=True)
data_bicycle_thefts.drop('Latitude', axis='columns', inplace=True)
data_bicycle_thefts.drop('ObjectId2', axis='columns', inplace=True)

# Changing Status value to 0 or 1
data_bicycle_thefts['Status'] = [1 if status=='RECOVERED' else 0 for status in data_bicycle_thefts['Status']]

# Checking if any null values exist in the cleaned data
print("Total Number of missing values: ", data_bicycle_thefts.isna().sum().sum())

# Store categorical values (column names) in an array
categorical_values = []
for col, col_type in data_bicycle_thefts.dtypes.iteritems():
    if col_type == 'O':
        categorical_values.append(col)
print(categorical_values)

# Use Label Encoder to convert categorical values to numeric
# Import label encoder 
from sklearn import preprocessing
# for each categorical column convert values to numeric
label_encoder = preprocessing.LabelEncoder()

mapping_dict = {}
for col in categorical_values:
    data_bicycle_thefts[col] = label_encoder.fit_transform(data_bicycle_thefts[col])
 
    le_name_mapping = dict(zip(label_encoder.classes_,
                               label_encoder.transform(label_encoder.classes_)))
 
    mapping_dict[col] = le_name_mapping
print(mapping_dict)


#Generate dataframe with correlation coefficients between columns
df_correlation = data_bicycle_thefts.corr()
print(df_correlation)

#Visualize correlation coefficients using the Seaborn library
sb.heatmap(df_correlation);

# Feature Selection: List of features that are most important for predictions as visualized in correlation chart
features = ["Occurrence_Hour", "Hood_ID", "Premises_Type", "Cost_of_Bike", "Status"]
featureSelection_df = data_bicycle_thefts[features]

# Store categorical values (column names) in an array
categorical_values = []
for col, col_type in featureSelection_df.dtypes.iteritems():
    if col_type == 'O':
        categorical_values.append(col)
print(categorical_values)

# Use Label Encoder to convert categorical values to numeric
# Import label encoder 
from sklearn import preprocessing
# for each categorical column convert values to numeric
label_encoder = preprocessing.LabelEncoder()

mapping_dict = {}
for col in categorical_values:
    featureSelection_df[col] = label_encoder.fit_transform(featureSelection_df[col])
 
    le_name_mapping = dict(zip(label_encoder.classes_,
                               label_encoder.transform(label_encoder.classes_)))
 
    mapping_dict[col] = le_name_mapping
print(mapping_dict)

for i in categorical_values:
    print(i)
    featureSelection_df[i]= label_encoder.fit_transform(featureSelection_df[i])

# Normalization/Standardization of the values with greater range to have same range
# Get column names without status
names = featureSelection_df.columns.difference(['Status'])
# Normalize
from sklearn.preprocessing import StandardScaler
cols_to_norm = names
scaled_df = featureSelection_df
scaled_df[cols_to_norm] = StandardScaler().fit_transform(featureSelection_df[cols_to_norm])


###################
#### BALANCE the IMBALANCED data here before splitting the data for training and testing
###################

# Split the data into train test
x = scaled_df[scaled_df.columns.difference(['Status'])]
y = scaled_df['Status']
y = y.astype(int)
y.value_counts()

# Show pie plot (Approach 1)
y.value_counts().plot.pie(autopct='%.2f')

# Undersampling (only for testing, but will go with oversampling as that 
# will have better accuracy as more records will be used to test)
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy=1) # Numerical value
# rus = RandomUnderSampler(sampling_strategy="not minority") # String
x_res, y_res = rus.fit_resample(x, y)

ax = y_res.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Under-sampling")

# Class distribution
y_res.value_counts()


# OVERSAMPLING
#Random Oversampling-  "not majority" = resample all classes but the majority class
from imblearn.over_sampling import RandomOverSampler

#ros = RandomOverSampler(sampling_strategy=1) # Float
ros = RandomOverSampler(sampling_strategy="not majority") # String
x_ros, y_ros = ros.fit_resample(x, y)

ax = y_ros.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Over-sampling")

# Class distribution
y_ros.value_counts()

# Split data: 80% for training and 20% for testing
from sklearn.model_selection import train_test_split
trainX,testX,trainY,testY = train_test_split(x_ros,y_ros, test_size = 0.2)

"""
3-4 Predictive model building & Model scoring and evaluation
"""
####################################################
########### 3. Predictive model building ###########
####################################################

########### LOGISTIC REGRESSION MODEL ###########
# Building logistic regression model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(x_ros, y_ros)

# Performing 10 fold cross validation to score the model (test accuracy)
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(lr, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print ('The score of the 10 fold run is: ',score)

testY_predict = lr.predict(testX)
testY_predict.dtype
print(testY_predict)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
labels = y.unique()
print(labels)
print("Accuracy:",metrics.accuracy_score(testY, testY_predict))
#Let us print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix \n" , confusion_matrix(testY, testY_predict, labels=labels))

#Use Seaborn heatmaps to print the confusion matrix in a more clear and fancy way
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(testY, testY_predict)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels([0, 1]); ax.yaxis.set_ticklabels([0, 1]);
plt.show()

# Plot ROC Curve for the Logistic Regression Model
from sklearn.metrics import roc_curve, roc_auc_score
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(testY, testY_predict)

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Logistic Regression Model')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

########### DECISION TREE MODEL ###########
# Build the decision tree using the training dataset. Use entropy as a method for splitting, 
# and split only when reaching 20 matches.

from sklearn.tree import DecisionTreeClassifier
dt_data = DecisionTreeClassifier(criterion='entropy',min_samples_split=20, random_state=99)
dt_data.fit(trainX, trainY)

#Test the model using the testing dataset and calculate a confusion matrix this time using pandas
predictions = dt_data.predict(testX)
pd.crosstab(testY,predictions,rownames=['Actual'],colnames=['Predictions'])

# 10 fold cross validation using sklearn 
from sklearn.model_selection import KFold
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=1)
from sklearn.model_selection import cross_val_score
score = np.mean(cross_val_score(dt_data, trainX, trainY, scoring='accuracy', cv=crossvalidation, n_jobs=1))
print('Score: ', score)
print("Accuracy:",metrics.accuracy_score(testY, predictions))

#Use Seaborn heatmaps to print the confusion matrix in a more clear and fancy way
cm = confusion_matrix(testY, predictions)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels([0, 1]); ax.yaxis.set_ticklabels([0, 1]);
plt.show()

# Plot ROC Curve for the Decision Tree Model
from sklearn.metrics import roc_curve, roc_auc_score
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(testY, predictions)

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - DecisionTree')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# We select the Decision Tree Model as it has high accuracy than the Logistic Regression Model
"""
5. Deploying Model
Serializing (saving) the model as an object
"""

import joblib
joblib.dump(dt_data, 'C:\COMP309-Group7-Project2\model_dt.pkl')
print("Model dumped!")


# Serializing and saving the model columns as an object
model_columns = list(x.columns)
print(model_columns)
joblib.dump(model_columns, 'C:\COMP309-Group7-Project2\model_columns.pkl')
print("Models columns dumped!")



















