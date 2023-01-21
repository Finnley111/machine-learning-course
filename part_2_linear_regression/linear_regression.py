# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv(r'C:\Users\finnl_y\OneDrive\Documents\GitHub\machine-learning-course\part_2_linear_regression\Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting test and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

###############################################
# Training the simple linear regression model
###############################################
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # the fit method trains the regression model on the data set

###############################################
# Predicting the test set results
###############################################
# y_pred is predicted salaries
# y_test is the actual colllected salaries
y_pred = regressor.predict(X_test) # predicts the salary of the test set

###############################################
# Visualizing the training set results
###############################################
plt.scatter(X_train, y_train, color = 'red')
# the y cordinate of this line is the predicted values of the training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # plots the regression line
plt.title('Salary vs Experience (Training set)') 
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

###############################################
# Visualizing the test set results
###############################################
plt.scatter(X_test, y_test, color = 'red')
# the regression line is from a unique equation so it doesnt matter if X_train is replaced with X_test
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()