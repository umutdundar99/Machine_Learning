# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 21:39:42 2022

@author: umut_
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler=pd.read_csv('Churn_Modelling.csv')

print(veriler)


X= veriler.iloc[:,3:13].values
Y= veriler.iloc[:,13].values


from sklearn import preprocessing


le=preprocessing.LabelEncoder()
X[:,1]=le.fit_transform(X[:,1])



le2=preprocessing.LabelEncoder()
X[:,2]=le2.fit_transform(X[:,2])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe= ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],remainder="passthrough")

X=ohe.fit_transform(X)
X=X[:,1:]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# Yapay Sinir Ağı


import keras

from keras.models import Sequential #Yapay Sinir Ağı oluşturur
from keras.layers import Dense      # Nöron oluşturabileceğimiz Nesne

# genelde girişte ve gizli katmanda relu kullanılır

classifier = Sequential()

classifier.add(Dense(6, activation="relu",input_dim=11)) # Giriş Katmanı

classifier.add(Dense(6,activation="relu")) # Gizli Katman 

classifier.add(Dense(1,activation="sigmoid")) # Çıkış katmanı

classifier.compile(optimizer='adam', loss= "binary_crossentropy", metrics=['accuracy'], )


classifier.fit(X_train,y_train, epochs=50) # X'den Y'yi öğren, epochs kaç kerede öğreneciğini belirler

y_pred=classifier.predict(X_test)

y_pred



y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

print(cm)




