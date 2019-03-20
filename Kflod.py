# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 22:39:29 2018

@author: SURAJ BHADHORIYA
"""
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import numpy as np
#make data set
x=np.array([1,2,3,4,5,6,7,8,9,10],dtype=np.float64)
y=np.array([11,12,13,14,15,16,17,18,19,20],dtype=np.float64)
# here divide into k folds
kf=KFold(n_splits=5,shuffle=True,random_state=2)
for x_val,y_val in kf.split(x):
    print(x_val,y_val)
    
#    apply model
reg=linear_model.LinearRegression()   
# divide it into x,y terain &x,y test
for x1,y1 in kf.split(x):
    print(x1,y1)
    x_train,x_test=x[x1],x[y1]
    y_train,y_test=y[x1],y[y1]
    print(x_train,y_train)
    print(x_test,y_test)
#    reshape x_train & x_test
 x_train=x_train.reshape(-1,1)
 x_test=x_test.reshape(-1,1)
 reg.fit(x_train,y_train)
 y_pred=reg.predict(x_test)
 print(y_pred)
 ac=reg.score(x_train,y_train)
 print("accuracy",ac)
 print("Error: ",mean_squared_error(y_test,y_pred))
 print("El r^2: ",r2_score(y_test,y_pred))