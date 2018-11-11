# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:33:21 2018

@author: prabhudayala
"""
import pickle
loaded_model = pickle.load(open('word2vec_on_amazon_food_vectors50.sav', 'rb'))
#result = w2v_model.wv.most_similar('tasty')
print(loaded_model.wv.most_similar('this'))

'''[('tastey', 0.9147965312004089), ('yummy', 0.8564854860305786), ('satisfying', 0.8361456394195557), ('filling', 0.8325173854827881), ('delicious', 0.8161024451255798), ('flavorful', 0.7939844727516174), ('tasteful', 0.7608951926231384), ('delectable', 0.7497645616531372), ('versatile', 0.7496802806854248), ('nutritious', 0.7485142350196838)]'''