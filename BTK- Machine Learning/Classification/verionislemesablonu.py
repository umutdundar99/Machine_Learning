# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

# X_Test için sadece transformu kullanıyoruz(fit eğit,transform o eğitimi uygula demek)
# X_test için bir daha öğrenmeden uygula demek için fit kullanmıyoruz.
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
#☺print(y_pred)
#print(y_test)


                # Confusion Matrix
                
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print(cm)


                # KNN Algorithm
from sklearn.neighbors import KNeighborsClassifier
# 1 farklı komşuya bakacak ve minkowski yönetimini kullanacak;
# default olarak n_neighbor=5'dir. 5 verdiğimizde daha hatalı sonuç verdi
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print('KNN')
print(cm)


                # SVC
                
from sklearn.svm import SVC
# linear,rbf gibi metodlar kullanılabilir.
svc = SVC(kernel='rbf') 
svc.fit(X_train,y_train) # X_train ile y_train arasında bağlantı kurarak öğren
y_pred = svc.predict(X_test)

cm=confusion_matrix(y_test, y_pred)
print('SVC')
print(cm)

                # Naiva Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train,y_train)

y_pred = gnb.predict(X_test)

cm=confusion_matrix(y_test,y_pred)

print("GNB")
print(cm)


                # Decision tree

from sklearn.tree import DecisionTreeClassifier

dtc= DecisionTreeClassifier(criterion='entropy')


dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)


                # Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier(n_estimators=10, criterion='entropy')

rfc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)



















