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
data_bicycle_thefts = pd.read_csv('C:/COMP309-Group7-Project/Bicycle_Thefts.csv')

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
data_bicycle_thefts.to_csv(r'C:/COMP309-Group7-Project/Bicycle_Thefts_2.0.csv', index=False, header=True)


"""
2. Data Modelling
"""
















