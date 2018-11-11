# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 13:09:58 2018

@author: prabhudayala
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.manifold import TSNE
df=pd.read_csv('./train.csv')
l=df['label']
#l=l.values.reshape(l.shape[0],1)
d=df.drop('label',axis=1)
print(d.shape)
print(l.shape)
#d=d.iloc[:5000]
#l=l.iloc[:5000]
l=l.values.reshape((l.shape[0],1))
print(d.shape)
print(l.shape)
model=TSNE(n_components=2,perplexity=30,random_state=0)
print("i will be busy")
tsne_data=model.fit_transform(d)
print("i am done")
tsne_data=np.hstack((tsne_data,l))
print(tsne_data.shape)
tsne_df=pd.DataFrame(data=tsne_data,columns=('firsteigen','secondeigen','label'))
sn.FacetGrid(tsne_df,hue='label',height=6).map(plt.scatter,'firsteigen','secondeigen').add_legend()
plt.show()