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

# X = np.matrix([[1,-5,2],[1,7,3],[1,11,-2],[1,15,-4]])
# y = np.array([2,3,4,5])

def predict(X,ps):
    hp = 0
    for i in range(X.shape[0]):
        print X[i],ps[i]
        hp = hp + X[i]*ps[i]
    return hp

def calc(X,params,y,i):
    r = np.dot(X,params.T) - y
    r = r*X[:,i]
    return r

def gradient_descent(X,y,pms,a):
    m,n = X.shape
    new_params = np.zeros(n)
    for i in range(n):
        x = (a/m)*calc(X,pms,y,i)
        new_params[i] = pms[i] - x
    return new_params

def linregr(X,y,a):
    m,n = X.shape
    params = np.zeros(n)
    for i in range(100):
        params = gradient_descent(X,y,params,a)
    return params

a=0.000005
ps = linregr(X,y,a)
# print ps

X1 = np.array([1,0.9,0.00,8.5,0,0.6,6,90,5,4,311,22,400,17])

hp = predict(X1,ps)
print hp
