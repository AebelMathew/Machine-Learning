# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:19:28 2022

@author: ACER
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from sklearn.datasets import load_iris
import scipy.spatial
from collections import Counter
data = load_iris()

x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y)
class KNN:
    def __init__(self, k):
        self.k = k
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    def distance(self, x1, x2):
        distance = scipy.spatial.distance.euclidean(X1, X2)
    def predict(self, x_test):
        final_output = []
        for i in range(len(x_test)):
            d = []
            votes = []
        for j in range(len(x_train)):
        dist = scipy.spatial.distance.euclidean(x_train[j] , x_test[i])
        d.append([dist, j])
        d.sort()
        d = d[0:self.k]
    for d, j in d:
    votes.append(y_train[j])
    ans = Counter(votes).most_common(1)[0][0]
    final_output.append(ans)
    return final_output

def score(self, X_test, y_test):
predictions = self.predict(X_test)
return (predictions == y_test).sum() / len(y_test)
model = KNN(20)
model.fit(x_train, y_train)

pred = model.predict(x_test)
print(model.score(x_test, y_test))
