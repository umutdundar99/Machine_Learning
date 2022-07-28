# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 20:05:35 2021

@author: umut_
"""

# Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
veriler=pd.read_csv("satislar.csv")
#print(veriler)

# veri ön işleme
aylar=veriler[['Aylar']]
satislar= veriler[['Satislar']]
print(satislar)
satislar2= veriler.iloc[:,:1].values
print(satislar2)

# verilerin ayrılması
from sklearn.model_selection import train_test_split
# aylar bağımsız değişken(x)
# satıslar cikisimiz yani bağımlı degisken(y)
x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33,random_state=6)


# verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler
"""
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)

Y_train= sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)

"""
# model oluşturma
# ctrl+ı tuşları ile kütüphaneler ve sınıflar hakkında bilgi alınabilir
# scale olarak kullandığımızda sonuçlar pek anlamlı gelmeyebilir. Scaling'i ortadan kaldırıp
# aşağıda X_test, Y_test vs. olmadan kullanıyoruz. İleriki örneklerde scaling gerekecek.
from sklearn.linear_model import LinearRegression

lr=LinearRegression() # linear regression sınıfından bir obje oluşturuyoruz
lr.fit(x_train,y_train)

tahmin=lr.predict(x_test) # tahmin ile y_test karşılaştırılabilir

# sort işlemi yapılmazsa sonuç anlamlı değil çünkü veriler sıralı değil
x_train= x_train.sort_index()
y_train= y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))