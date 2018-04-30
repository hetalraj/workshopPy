# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:17:49 2018

@author: Helat
"""

from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load the iris datasets
dataset = datasets.load_iris()
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
X=dataset.data
Y=dataset.target
rfe = rfe.fit(X, Y)
# summarize the selection of the attributes

print(rfe.ranking_)