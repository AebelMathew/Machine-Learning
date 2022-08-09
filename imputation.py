import pandas as pd
from sklearn.impute import SimpleImputer



data = pd.read_csv('D:\AEBEL MATHEW\STUDIES\SEMS\WINTER SEM 2021-22\C1-CSE4020-ML\LAB\DATASETS\MELBOURNE_HOUSE_PRICES_LESS.csv')

print("Dimension of the dataset: ",data.shape)

na_variables = [ var for var in data.columns if data[var].isnull().mean() > 0 ]
print("Columns of the dataset containing null values: ",na_variables)

na_variables_mean=data[na_variables].mean()
print("Means of columns of the dataset containing null values: ",na_variables_mean)

na_count=data[na_variables].isnull().sum()
print("No. of rows of columns of the dataset containing null values: ",na_count)

frequent_impute=data[na_variables].fillna(data[na_variables].mean())

print("Data stored in the columns with null values:")
print(data[na_variables])

print("Data stored in the columns after replacing null values:")
print(frequent_impute)
