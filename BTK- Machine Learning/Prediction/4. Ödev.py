# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 19:45:36 2021

@author: umut_
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# upload data
veriler=pd.read_csv("odev_tenis.csv")
#print(veriler)
degiskenler=veriler.iloc[:,1:3].values
                         
# katagorik veriler nümerik verilere dönüştürülecek


# outlook
from sklearn import preprocessing
outlook = veriler.iloc[:,0:1].values
le=preprocessing.LabelEncoder()
outlook[:,0]=le.fit_transform(veriler.iloc[:,0])

ohe=preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()


# windy
from sklearn import preprocessing
windy= veriler.iloc[:,3:4].values
le=preprocessing.LabelEncoder()
windy[:,0]=le.fit_transform(veriler.iloc[:,-2])
ohe=preprocessing.OneHotEncoder()
windy=ohe.fit_transform(windy).toarray()
windy=windy[:,1]

# play
from sklearn import preprocessing
play=veriler.iloc[:,-1:].values
le=preprocessing.LabelEncoder()
play[:,0]=le.fit_transform(veriler.iloc[:,-1])



# numpy dizilerini pandas dataframe'ine dönüştürüyoruz

outlook_dataframe= pd.DataFrame(data=outlook,index=range(14),columns=['overcast','rainy','sunny'])

windy_dataframe= pd.DataFrame(data=windy,index=range(14),columns=['windy'])

temparature_dataframe=pd.DataFrame(data=degiskenler[:,0],index=range(14),columns=['temperature'])

humidity_dataframe=pd.DataFrame(data=degiskenler[:,1],index=range(14),columns=['humidity'])

play_dataframe= pd.DataFrame(data=play,index=range(14),columns=['play'])

s=pd.concat([outlook_dataframe,temparature_dataframe],axis=1)

s1=pd.concat([windy_dataframe,s],axis=1)

x_verisi=pd.concat([play_dataframe,s1],axis=1)

concat=pd.concat([x_verisi,humidity_dataframe],axis=1)
# veri bölme

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_verisi,humidity_dataframe,test_size=0.33,random_state=0)

# öznitelik ölçekleme

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train) # x_train'den y_train'i öğren

y_pred = regressor.predict(x_test)


# BACKWARD ELEMINATION
play=concat.iloc[:,-1:].values

import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int),values=humidity_dataframe,axis=1) 
X_l=concat.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)
model= sm.OLS(humidity_dataframe,X_l).fit()
print(model.summary())

concat=concat.iloc[:,[0,2,3,4,5]]

import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int),values=humidity_dataframe,axis=1) 
X_l=concat.iloc[:,[0,1,2,3,4]].values
X_l=np.array(X_l,dtype=float)
model= sm.OLS(humidity_dataframe,X_l).fit()
print(model.summary())


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

x_train = x_train.iloc[:,[0,2,3,4,5]]
x_test = x_test.iloc[:,[0,2,3,4,5]]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)



