# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 18:09:04 2018

@author: prabhudayala
"""

def cleanhtml(sentence): #function to clean the word of any html-tags\n",
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters\n",
    cleaned = re.sub(r'[?|!|\\'|\"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\\|/]',r' ',cleaned)
    return  cleaned
print(stop)
print('************************************')
print(sno.stem('tasty'))









i=0
str1=' '
final_string=[],
all_positive_words=[] # store words from +ve reviews here\n",
all_negative_words=[] # store words from -ve reviews here.\n",
s='',
for sent in tqdm(final['Text'].values):
    filtered_sentence=[]
    #print(sent);\n",
    sent=cleanhtml(sent) # remove HTMl tags\n",
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                        filtered_sentence.append(s)
                        if (final['Score'].values)[i] == 'positive'
                            all_positive_words.append(s) #list of all words used to describe positive reviews\n",
                        if(final['Score'].values)[i] == 'negative'
                            all_negative_words.append(s) #list of all words used to describe negative reviews reviews\n",
                else:
                continue
            else:
                continue
            #print(filtered_sentence)
            str1 = b" ".join(filtered_sentence) #final string of cleaned words
            #print(\"***********************************************************************\")
    
            final_string.append(str1)
            i+=1
    
    '''"    #############---- storing the data into .sqlite file ------########################\n",
    "    final['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review \n",
    "    final['CleanedText']=final['CleanedText'].str.decode(\"utf-8\")\n",
    "        # store final table into an SQlLite table for future.\n",
    "    conn = sqlite3.connect('final.sqlite')\n",
    "    c=conn.cursor()\n",
    "    conn.text_factory = str\n",
    "    final.to_sql('Reviews', conn,  schema=None, if_exists='replace', \\\n",
    "                 index=True, index_label=None, chunksize=None, dtype=None)\n",
    "    conn.close()\n",