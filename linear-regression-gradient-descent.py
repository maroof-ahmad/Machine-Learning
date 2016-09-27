'''
Linear regression using gradient descent
Author-Maroof Ahmad
Dataset used - Boston House Prices, https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("housing.data.txt", delim_whitespace = True)
df.insert(0,"THETA",1)
y = np.matrix(df["MEDV"])
X = np.matrix(df.drop("MEDV",axis=1))

def predict(X,ps):
    x = X[np.newaxis]
    return np.dot(X,ps)


def linregr(X,y,a):
    m,n = X.shape
    y = y[np.newaxis]
    y = y.T
    params = np.zeros(n)
    params = params[np.newaxis]
    params = params.T
    for i in range(1000):
        h = np.dot(X,params)
        h = h - y
        g = np.dot(X.T,h)/m
        params = params - a*g
    return params

a=0.000005
ps = linregr(X,y,a)
print ps

X1 = np.array([1,0.9,0.00,8.5,0,0.6,6,90,5,4,311,22,400,17])

hp = predict(X1,ps)
print hp[0][0]
