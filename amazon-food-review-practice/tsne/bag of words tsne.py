# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 19:38:21 2018

@author: prabhudayala
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sn

con=sqlite3.connect('./final.sqlite')

filtered_data=pd.read_sql_query('''
select * 
from Reviews''',con)

#print(filtered_data.iloc[1].values)
#print(filtered_data['CleanedText'].values[10])
#print(filtered_data['Score'].values.shape)

from sklearn.feature_extraction.text import CountVectorizer
count_vect=CountVectorizer()
final_counts=count_vect.fit_transform(filtered_data['CleanedText'].values)
#print(final_counts.get_shape())

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=5, n_iter=7)
#abc.transform(final_counts)
#d=svd.fit(final_counts)
d= svd.fit_transform(final_counts)
print(d.shape)
l=filtered_data['Score']
d=d[:10000]
d=pd.DataFrame(d)
l=l.iloc[:10000]
l=l.values.reshape((l.shape[0],1))
print(d.shape)
print(l.shape)


from sklearn.manifold import TSNE
model=TSNE(n_components=2,perplexity=50,random_state=0)
print("i will be busy")
tsne_data=model.fit_transform(d)
print("i am done")
tsne_data=np.hstack((tsne_data,l))
print(tsne_data.shape)
tsne_df=pd.DataFrame(data=tsne_data,columns=('firsteigen','secondeigen','label'))
sn.FacetGrid(tsne_df,hue='label',height=6).map(plt.scatter,'firsteigen','secondeigen').add_legend()
plt.show()