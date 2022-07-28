# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 00:01:23 2021

@author: umut_
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
# Veri Yükleme
veriler=pd.read_csv('maaslar.csv')


# data frame dilimleme (slice)
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,-1:]


# NumPy array transform
X=x.values
Y=y.values

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y) # x'den y'yi öğren
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

# Non-linear Model
# 2nd degree Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y) 
# x_poly'yi y'ye göre fit et(öğren)
# Aslında çarpanları öğreniyor
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg2.predict(x_poly),color='blue')
plt.show()


                                            # 4th degree Polynomial Regression
poly_reg3= PolynomialFeatures(degree=4)
x_poly3= poly_reg3.fit_transform(X)
lin_reg3= LinearRegression()
lin_reg3.fit(x_poly3,y)
# x_poly'yi y'ye göre fit et(öğren)
# Aslında çarpanları öğreniyor


# Visualization
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color='blue')
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg2.predict(x_poly),color='blue')
plt.show()


plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg3.predict(x_poly3),color='blue')
plt.show()

#Predictions
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

print('Polynomial R2 Değeri')
print(r2_score(Y, lin_reg3.predict(poly_reg3.fit_transform(X))))
                                    

# verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)


                                    # Support Vector Kullanımı
                                    
                                    
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()
#print(svr_reg.predict([[11]]))
#print(svr_reg.predict([[6.6]]))

print('SVR R2 Değeri')
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

                                    # Karar Ağaçları ile Tahmin
Z=X+0.5
K=X-0.4
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)

r_dt.fit(X,Y)

plt.scatter(X,Y,color="pink")
plt.plot(X,r_dt.predict(X),color="blue")
plt.show()
# Decision Tree var olan sonuçları döndürebilir, yeni sonuçlar üretmiyor.
#print(r_dt.predict([[11]])) # 9'dan sonraki herkesi 50'ye sabitledi
#print(r_dt.predict([[6.6]])) # 6'dan sonrası için herkesi 10'a sabitledi ki yanlış sonuç
print('Decision Tree R2 Değeri')
print(r2_score(Y,r_dt.predict(X)))

                                    # Random Forest

from sklearn.ensemble import RandomForestRegressor

# n_estimator decision tree sayısını belirler. 10 farklı küçük küme oluşturulur
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0) # obje oluşturduk
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.5]]))

plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="blue")

plt.plot(X,rf_reg.predict(Z),color="green")
plt.plot(X,r_dt.predict(K),color="yellow")


                                    # R Square ile Hata Oranı Bulma

from sklearn.metrics import r2_score
print('Random Forest R2 Değeri')
print(r2_score(Y,rf_reg.predict(X)))











