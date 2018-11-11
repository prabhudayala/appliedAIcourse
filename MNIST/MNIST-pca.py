# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 12:48:52 2018

@author: prabhudayala
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('./train.csv')
l=df['label']
#l=l.values.reshape(l.shape[0],1)
d=df.drop('label',axis=1)
print(d.shape)
print(l.shape)

#plotting the image from data matrix
plt.figure(figsize=(3,3))
index=5989
data=d.iloc[index]
#reshapes the data to desired matrix size
data=data.values.reshape(28,28)
print(data.shape)
#shows the image
plt.imshow(data,interpolation='None',cmap='gray')
plt.show()
print(l[index])


from sklearn.preprocessing import StandardScaler
std_data= StandardScaler().fit_transform(d)
print(std_data.shape)
sample_data=std_data
cov_matrix=np.dot(sample_data.T,sample_data)
print(cov_matrix.shape)


from scipy.linalg import eigh
values,vectors=eigh(cov_matrix,eigvals=(782,783))
vectors=vectors.T
print(vectors.shape)
print(sample_data.shape)
new_coordinates=np.dot(vectors,sample_data.T)
print(new_coordinates.shape)
print(l.shape)
new_coordinates=np.vstack((new_coordinates,l)).T

print(new_coordinates.shape)
df1=pd.DataFrame(data=new_coordinates,columns=('firsteigen','secondeigen','label'))
#print(df1.head)

import seaborn as sn
sn.FacetGrid(df1,hue='label',size=6).map(plt.scatter,'firsteigen','secondeigen').add_legend()
plt.show()
from sklearn import decomposition
pca=decomposition.PCA(n_components=2)
pca_data=pca.fit_transform(d)
print(pca_data.shape)




pcaForDimentionalityreduction=decomposition.PCA(n_components=784)
pca_data1=pcaForDimentionalityreduction.fit_transform(d)
percentageVarianceExplained=pcaForDimentionalityreduction.explained_variance_/np.sum(pcaForDimentionalityreduction.explained_variance_)
cum_variance_explained=np.cumsum(percentageVarianceExplained)
#print (pcaForDimentionalityreduction.explained_variance_)
#print (pcaForDimentionalityreduction.explained_variance_ratio_)
#print (pcaForDimentionalityreduction.explained_variance_ratio_.cumsum())
plt.clf()
plt.plot(cum_variance_explained,linewidth=2)
plt.show()