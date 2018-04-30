# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:09:03 2018

@author: Helat
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
dataset=load_boston()

#//Object boston is a dictionary ,so we can explore the keys of this object

dataset.keys()
#the columns headers are numbers
#converting the dataset to a dataframe

boston=pd.DataFrame(dataset.data)

#Checking the dimensions

boston.shape

print(dataset.feature_names)
#giving vcolumns names instaed of numbers to columns

boston.columns=dataset.feature_names
print(boston.head())

print(dataset.target) ##dataset.target has house prices
boston['PRICE']=dataset.target #adding a new columns 'PRICE' to the dataframe
#Now lets use a linear model and predict the house prices for new sets of data
boston.head()
#Lets separate our class labels from the data
Y=boston.iloc[:,13]
X=boston.drop('PRICE',axis=1)

##Feature selection 

from sklearn import datasets
from sklearn.feature_selection import RFE


model = LinearRegression() # create a base classifier used to evaluate a subset of attributes
rfe = RFE(model, 3) # create the RFE model and select 3 attributes

rfe = rfe.fit(X, Y)
print(rfe.ranking_) # summarize the selection of the attributes
featureranks=rfe.ranking_
print("Ranks of features",featureranks)

for i in featureranks:
    print i
    
pd.DataFrame(list(zip(X.columns,featureranks)),columns=['features','Ranks'])   
    

lm=LinearRegression()
lm.fit(X, Y)
print('Estimated intercept coefficient',lm.intercept_)
print(' Number of coefficients:',len(lm.coef_))

pd.DataFrame(list(zip(X.columns,lm.coef_)),columns=['features','estimatedCoefficients'])

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=5)

print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape

lm.fit(X_train,Y_train)

pred_train=lm.predict(X_train)

pred_test=lm.predict(X_test)

##calculate the MEan squared error
##error on train data
train_error=np.mean((Y_train-(pred_train))**2)
##error on test data
test_error=np.mean((Y_test-(pred_test))**2)

print("Error on training data",train_error)


print("Error on testing data",test_error)

plt.scatter(pred_train,pred_train-Y_train,c='b',label="train")
plt.scatter(pred_test,pred_test-Y_test,c='g',label="test")
plt.hlines(y=0,xmin=0,xmax=50)
plt.title("Residual plot with train(blue) and test data(green)")
plt.legend(loc="upper left")
plt.ylabel('Residuals')
plt.xlabel('scores')


