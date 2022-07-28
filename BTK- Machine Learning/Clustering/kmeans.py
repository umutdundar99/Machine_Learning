# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 22:59:11 2022

@author: umut_
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv('musteriler.csv')

X = veriler.iloc[:,2:4].values


                        #KMeans
from sklearn.cluster import KMeans

kmeans= KMeans(n_clusters=3, init='k-means++')
kmeans.fit(X)
print(kmeans.cluster_centers_) # cluster centers'i verir.
sonuclar=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_) # k-meansin ne kadar başarılı olduğu, wcss değeridir

#plt.plot(range(1,11),sonuclar)


kmeans = KMeans(n_clusters=3,init='k-means++',random_state=123)
Y_tahmin= kmeans.fit_predict(X)
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100,c='red') # X'in y tahmini 0 ise 0 al,
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100,c='green')
plt.title('Kmeans')
plt.show()


                        # Agglomerative
from sklearn.cluster import AgglomerativeClustering

ac=AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
Y_tahmin=ac.fit_predict(X) # hem inşa et hem tahmin et
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100,c='red') # X'in y tahmini 0 ise 0 al,
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100,c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100,c='green')
plt.title('HC')
plt.show()
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()

