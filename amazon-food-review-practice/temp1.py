# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 11:39:19 2018

@author: prabhudayala
"""

import sqlite3
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

#for bag of words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

con=sqlite3.connect('./database.sqlite')

filtered_data=pd.read_sql_query('''
select * 
from Reviews
where Score!=3
                                ''',con)
#print(filtered_data.head())

def partition(x):
    if x<3:
        return 'negative'
    return 'positive'

actualScore=filtered_data['Score']
positiveNegative=actualScore.map(partition)
filtered_data['Score']=positiveNegative

print(filtered_data.shape)


sorted_data=filtered_data.sort_values('ProductId',axis=0)
final=sorted_data.drop_duplicates(subset={'UserId','ProfileName','Time','Text'})
print(final.shape)

final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
print(final.shape)
print(final['Score'].value_counts())


#text preprocessing
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
stop=set(stopwords.words('english'))
sno = nltk.stem.SnowballStemmer('english')

def cleanhtml(sentence): #function to clean the word of any html-tags\n",
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters\n",
    cleaned = re.sub(r'[?|!|\'|\"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\\|/]',r' ',cleaned)
    return  cleaned

print(stop)
print(sno.stem('beautiful'))
    
i=0
str1=' '
final_string=[]
all_positive_words=[] # store words from +ve reviews here\n",
all_negative_words=[] # store words from -ve reviews here.\n",
s=''
for sent in final['Text'].values:
    filtered_sentence=[]
    #print(sent);\n",
    sent=cleanhtml(sent) # remove HTMl tags\n",
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    #print(i)
                    if (final['Score'].values)[i] == 'positive':
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue
            #print(filtered_sentence)
            str1 = b" ".join(filtered_sentence) #final string of cleaned words
            #print(\"***********************************************************************\")
    
    final_string.append(str1)
    i+=1
#
#print(final_string)
print(len(final_string))
final['CleanedText']=final_string
print(final['CleanedText'])
print(final.head(3))