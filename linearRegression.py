# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 19:48:36 2020

@author: Bonga
"""

import matplotlib.pyplot as plt
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn import linear_model

d = load_diabetes()
d_X =d.data[:, np.newaxis , 2]
dx_train = d_X[:-20]
dy_train = d.target[:-20]
dx_test = d_X[-20:]
dy_test = d.target[-20:]

#lr = linear_model.LinearRegression() #replace with functions


# y =mx+b
# m formula
# b formula
# use numpy to calc GRADIENT(squeeze function ?)
# train data without lr.fit (m,b)
# predict dx_test without lr

def fit(x,y):
    m = ((np.mean(x) * np.mean(y)) - np.mean(x * y))/((np.mean(x))**2 - np.mean(x**2))
    b = np.mean(y) - m * np.mean(x)
    return m,b

m,b = fit(np.squeeze(dx_train), dy_train)


plt.scatter(dx_test,dy_test)
plt.plot([np.min(dx_train),np.max(dx_train)],[np.min(dx_train)*m + b, np.max(dx_train)*m + b], c='r')
plt.show()



#plt.plot(dx_test, lr.predict(dx_test),c='r') 


