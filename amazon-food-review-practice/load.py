# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:46:21 2018

@author: prabhudayala
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:41:57 2018

@author: prabhudayala
"""

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
import pandas as pd
tfidf_sent_vectors = pickle.load(open('tfidf_word2vec_on_amazon_food_vectors50.sav', 'rb'))
#print(len(tfidf_sent_vectors))
#print(tfidf_sent_vectors[0])
d=pd.DataFrame(tfidf_sent_vectors[:10000])
l=filtered_data['Score']
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