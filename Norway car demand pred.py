# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 12:16:48 2023

@author: eramt
"""


import pandas as pd
import os
import numpy as np
import wooldridge as woo
import statsmodels.formula.api as smf
import math
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
from linearmodels.panel import compare
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor 



# The block of code below auto-clears the console each time you run the program 
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass
import datetime as dt
from datetime import datetime

# Auto clears plots from previous runs
f = plt.figure()
f.clear()
plt.close(f)

# The option below will force the print command to display all of the columns requested 
pd.set_option('display.max_columns', None)


os.chdir('C:/Users/eramt/OneDrive - Washington State University (email.wsu.edu)/Machine Learning/ML Project/Practice Demand Forecasting')


# Load the CSV file (should be in the same directory) 
data = pd.read_csv('norway_new_car_sales_by_make1.csv') 

# Data Pre-processing:

# Create a column “Period” with both the Year and the Month 
data['Period'] = data['Year'].astype(str) + '-' + data['Month'].astype(str) 
# We use the datetime formatting to make sure format is consistent 
data['Period'] = pd.to_datetime(data['Period']).dt.strftime('%Y-%m') 
 
# Create a pivot of the data to show the periods on columns and the car makers on rows 
df = pd.pivot_table(data=data, values='Quantity', index='Make', columns='Period', aggfunc='sum', fill_value=0) 
 
# Print data to Excel for reference 
df.to_excel('Clean Demand.xlsx')


#Create a function datasets that takes as inputs:

#df our initial historical demand;
#x_len the number of months we will use to make a prediction;
#y_len the number of months we want to predict;
#y_test_len the number of months we leave as a final test;
#and returns X_train, Y_train, X_test & Y_test.



def datasets(df, x_len=12, y_len=1, y_test_len=12):
    D = df.values
    periods = D.shape[1]

    # Training set creation: run through all the possible time windows
    loops = periods + 1 - x_len - y_len - y_test_len 
    train = []
    for col in range(loops):
        train.append(D[:,col:col+x_len+y_len])
    train = np.vstack(train)
    X_train, Y_train = np.split(train,[x_len],axis=1)

    # Test set creation: unseen “future” data with the demand just before
    max_col_test = periods - x_len - y_len + 1
    test = []
    for col in range(loops, max_col_test):
        test.append(D[:,col:col+x_len+y_len])
    test = np.vstack(test)
    X_test, Y_test = np.split(test,[x_len],axis=1)

    # this data formatting is needed if we only predict a single period
    if y_len == 1:
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()

    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = datasets(df)

#Max depth Maximum amount of consecutive questions (nodes) the tree can ask.
#Min samples split Minimum amount of samples that are required in a node to trigger a new split. If you set this to 6, a node with only 5 observations left won’t be split further.
#Min samples leaf Minimum amount of observations that need to be in a leaf. This is a very important parameter. The closer this is to 0, the higher the risk of overfitting, as your tree will actually grow until it asks enough questions to treat each observation separately.
#Criterion This is the KPI that the algorithm will minimize (either MSE or MAE).

#Depending on your dataset, you might want to give different values to these parameters.

 
# — Instantiate a Decision Tree Regressor 
tree = DecisionTreeRegressor(max_depth=5,min_samples_leaf=5) 
 
# — Fit the tree to the training data 
tree.fit(X_train,Y_train)


# We created a tree with a maximum depth of 5 (i.e., a maximum of five yes/no consecutive questions are asked to classify one point)
# Each tree leaf has at minimum 5 samples.

# We now have a tree trained to our specific demand history. 
# We can already measure its accuracy on the training dataset.


# Create a prediction based on our model 
Y_train_pred = tree.predict(X_train) 
 
# Compute the Mean Absolute Error of the model 

MAE_tree = np.mean(abs(Y_train - Y_train_pred))/np.mean(Y_train) 
 
# Print the results 
print(f'Tree on train set MAE%: {round(MAE_tree*100,1)}\n')


Y_test_pred = tree.predict(X_test) 
MAE_test = np.mean(abs(Y_test - Y_test_pred))/np.mean(Y_test) 
print(f'Tree on test set MAE%: {round(MAE_test*100,1)}\n')

# This means that our regression tree is overfitted to the historical demand: 
# We lost 6 points of MAE in the test set compared to the historical dataset.


# Going further
#   There are many ways to improve this result further:
#  - Optimize the tree parameters.
#  - Use more advanced models (like a Forest, ETR, Extreme Gradient Boosting).
#  - Optimize the input data.
#  - Use external data.



# Using Random forest for same

from sklearn.ensemble import RandomForestRegressor 

#Instantiate a Random Forest Regressor 
forest = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5) 
 
# Fit the forest to the training data 
forest.fit(X_train,Y_train)

# Create a prediction based on our model 
Y_train_pred_random = forest.predict(X_train) 
 
# Compute the Mean Absolute Error of the model 

MAE_random = np.mean(abs(Y_train - Y_train_pred_random))/np.mean(Y_train) 
 
# Print the results 
print(f'Random forest on train set MAE%: {round(MAE_tree*100,1)}\n')


Y_test_pred_random = forest.predict(X_test) 
MAE_test = np.mean(abs(Y_test - Y_test_pred_random))/np.mean(Y_test) 
print(f'Random forest on test set MAE%: {round(MAE_test*100,1)}\n')




print(data.head())














