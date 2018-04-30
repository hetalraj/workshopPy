# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:32:26 2018

@author: Helat
"""

import sklearn
from sklearn import datasets
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier,NearestNeighbors

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)
X=dataset.iloc[:,0:4]
X.drop(X.index[1], inplace=True)
Y=dataset.iloc[:,4]
Y.drop(Y.index[1],inplace=True)
nn=NearestNeighbors(5)
nn.fit(X,Y)
import numpy as np
test=np.array([4.9,3.0,1.4,0.2])
test1=test.reshape(1,-1)
print(nn.kneighbors(test1,5))
#dataset.iloc([],)
#dataset.ix([64, 59, 98, 88, 61],)
KNN=KNeighborsClassifier(5)
KNN.fit(X,Y)
predictedclass=KNN.predict(test1)
print(predictedclass)