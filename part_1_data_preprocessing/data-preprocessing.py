import numpy as np # allows us to work with arrays
import matplotlib.pyplot as plt # plotting
import pandas as pd # creating matrices

##########################################
# Importing the dataset
##########################################
dataset = pd.read_csv(r'C:\Users\finnl_y\OneDrive\Documents\GitHub\machine-learning-course\part_1_data_preprocessing\Data.csv') # creates a dataframe var
X = dataset.iloc[:, :-1].values # features/independent variables
# iloc means locate indices. iloc[rows, cols].values.
# .values takes the values inside the cells
y = dataset.iloc[:, -1].values # dependent variable, what we want to predict using the dependent variables

print('\n', X)

##########################################
# Taking care of missing data
##########################################
from sklearn.impute import SimpleImputer
# np.nan means we want to replace missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
# looks at missing values and computes mean
# argumetns are all columns with numerical value, this way it returns all missing data
imputer.fit(X[:, 1:3])
# applies the transformation and adds the missing values
X[:, 1:3] = imputer.transform(X[:, 1:3])

print('\n', X)

##########################################
# Encoding the categorical data
##########################################

# The dependent variable
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder # use these for non-binary outcome (more than two categories)
# column transformer args
# 1. transformers = [(transformation_type, the name of the class that does the transformation, the indexes that are being transformed)]
# 2. remainder = 'passthrough' means to ignore the other colums
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print('\n', X)

# The independent variable
from sklearn.preprocessing import LabelEncoder # use this for binary outcome (two categories)
le = LabelEncoder()
y = le.fit_transform(y)

print('\n', y)

##########################################
# Splitting the dataset into the training set and test set
##########################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print('\n', X_train)
print('\n', X_test)
print('\n', y_train)
print('\n', y_test)


# feature scaling makes all of the variables considered equally so one of the vars doesnt dominate the others
# apply feature scaling after splitting the dataset into the training and test set
# this is because the test set is supposed to be brand new to the model and we want to see how the model would work on new data
# only need to be applied if we have features taking very different values
# using standardisation is more general to use than normalisation so use standardisation more often
# stadardisation produces values from -3 to +3
##########################################
# Feature scaling
##########################################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:]) # gets the mean and standard deviation and transforms it
X_test[:, 3:] = sc.transform(X_test[:, 3:]) # applies the transformation to the test matrix

print('\n', X_train)

print('\n', X_test)