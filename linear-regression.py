'''
Linear regression using normal equation
Author-Maroof Ahmad
Dataset used - Boston House Prices, https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def linregr(X,y):
    try:
        params = np.dot(np.dot( np.linalg.inv(np.dot(X.T,X)), X.T), y.T)
    except:
        params = np.dot(np.dot( np.linalg.pinv(np.dot(X.T,X)), X.T), y.T)
    pms = np.array([param for param in np.nditer(params)])
    return pms

def predict(X,ps):
    hp = 0
    for i in range(X.shape[0]):
        print X[i],ps[i]
        hp = hp + X[i]*ps[i]
    return hp


df = pd.read_csv("housing.data.txt", delim_whitespace = True)
df.insert(0,"THETA",1)
y = np.matrix(df["MEDV"])
X = np.matrix(df.drop("MEDV",axis=1))

ps = linregr(X,y)
for p in ps:
    print p
X1 = np.array([1,0.9,0.00,8.5,0,0.6,6,90,5,4,311,22,400,17])

hp = predict(X2,ps)
print hp
