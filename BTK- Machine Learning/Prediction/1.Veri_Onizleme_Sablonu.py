# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# ders 6: kutuphanelerin yuklenmesi
# import section
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# codes
# data upload
veriler= pd.read_csv('eksikveriler.csv') # comma seperated value
print(veriler)
boy=veriler[["boy"]]
print(boy)

                    # missing values

#sci-kit learn
from sklearn.impute import SimpleImputer
# nan değerler yerine ortalama eklenmesini söylüyoruz
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
yas=veriler.iloc[:,1:4].values
print(yas)
imputer=imputer.fit(yas[:,1:4]) # değerlerin değişirilmesini öğrenir(mean), 1'den 4. kolona kadar öğren demek.
# stratejiyi öğrenir.
yas[:,1:4]=imputer.transform(yas[:,1:4]) # değerleri değiştirir
print(yas)
                # katagorik veriler
ulke=veriler.iloc[:,0:1].values
print(ulke)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

sonuc=pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)
sonuc2=pd.DataFrame(data=yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet= veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)

s = pd.concat([sonuc,sonuc2],axis=1) # axis=1 ile yanyana yazıldı
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

                 # veri bölme (train-test)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33,random_state=0)

                # Öznitelik Ölçekleme
                # farklı dünyalardaki verileri aynı türe çevirip ölçekledik
from sklearn.preprocessing import StandardScaler

sc=StandardScaler() # sc bir objedir
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

