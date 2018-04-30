# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:57:05 2018

@author: Helat
"""

import sklearn
from sklearn import datasets
from __future__ import division  # only for Python 2
from sklearn import datasets
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

wine = datasets.load_wine()

features = wine.data
labels = wine.target

# split the data into training and testing
train_feats, test_feats, train_labels, test_labels = tts(features, labels, test_size=0.2)



# training
clf.fit(train_feats, train_labels)

# predictions
predictions = clf.predict(test_feats)
print "\nPredictions:", predictions

score = 0
for i in range(len(predictions)):
    if predictions[i] == test_labels[i]:
        score += 1
print "Accuracy:", (score / len(predictions)) * 100, "%"

# or, just do this for accuracy
# print accuracy_score(test_labels, predictions)