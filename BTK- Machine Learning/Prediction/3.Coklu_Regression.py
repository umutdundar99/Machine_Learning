# -*- coding: utf-8 -*-

# import section
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
# codes
# data upload
veriler= pd.read_csv('veriler.csv') # comma seperated value
print(veriler)
Yas=veriler.iloc[:,1:4].values
                # katagorik veriler --> nümerik
ulke=veriler.iloc[:,0:1].values
print(ulke)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)
ohe=preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

                # katagorik veriler --> nümerik              
c=veriler.iloc[:,-1:].values
print(c)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(c)
       


# numpy dizileri dataframe donusumu
sonuc= pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
print(sonuc)
sonuc2= pd.DataFrame(data=Yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet= veriler.iloc[:,-1].values
print(cinsiyet)

# dummy variable'den kurtulmak için sadece bir kolonu aldık
sonuc3=pd.DataFrame(data=c[:,:1],index=range(22),columns=['cinsiyet'])
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
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train) # x_train'den y_train'i öğren

y_pred = regressor.predict(x_test) # öğrendiğin şeyi tahmin et ve y_pred'e yaz


boy= s2.iloc[:,3:4].values
#print(boy)

# sol=s2.iloc[:,:3] # bütün satırları al ve 3'e kadar kolonları al
# sag=s2.iloc[:,4:] # boy kolonunun sağı ve solunu aldık, boy kolonu gitmiş oldu.

# veri=pd.concat([sol,sag],axis=1) # bu kodlar yerine aşağıdaki kod yazılabilir

veri=s2.drop("boy",axis=1)


                            # BOYU TAHMİN ETME
# veriyi yeniden böldük ve bu sefer y=boy oldu.
x_train,x_test,y_train,y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)

regressor2 = LinearRegression()

regressor2.fit(x_train,y_train)

y_pred=regressor2.predict(x_test)


                            # BACKWARD ELEMINATION
                         
import statsmodels.api as sm

# verinin başnına veri boyunda kolon olarak 1'lerden oluşan numpy dizisi ekledik  
X = np.append(arr=np.ones((22,1)).astype(int),values=veri,axis=1)         

X_l=veri.iloc[:,[0,1,2,3,4,5]].values # tek tek belirtmemizin sebebi ileride buradan index çıkaracak olmamız
X_l=np.array(X_l,dtype=float)
# bulmak istediğimiz değer boy1
model= sm.OLS(boy,X_l).fit()
print(model.summary()) # P değeri ne kadar küçükse bizim için o kadar iyi P>|t|, x5'i elemeliyiz

# X5 kaldırıldı

X_l=veri.iloc[:,[0,1,2,3,5]].values 
X_l=np.array(X_l,dtype=float)
model= sm.OLS(boy,X_l).fit()
print(model.summary())


# zorunlu değil ama son veriyi de kaldırdık
X_l=veri.iloc[:,[0,1,2,3]].values 
X_l=np.array(X_l,dtype=float)
model= sm.OLS(boy,X_l).fit()
print(model.summary())





