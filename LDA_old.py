
# coding: utf-8

# In[2]:

get_ipython().magic(u'pylab inline')
#!/usr/bin/env python
# -*- coding: utf-8 -*-


# In[3]:

import pandas as pd
import re
import numpy as np
from collections import Counter
from ast import literal_eval
from gensim import corpora, models
from nltk.stem.snowball import FrenchStemmer


# In[4]:

def clean(s):
    return s.decode('utf-8').replace('(','').replace(')','').replace('[','').replace(']','').replace('{','').replace('}','').replace("l'",'').replace("d'",'').replace(':','').replace(';','').replace(',',' ').replace('-','').replace('.',' ').replace('"','').replace('\t','').replace('!','').replace("c'",'').replace('\n','')
stemmer = FrenchStemmer()

def stem(doc,stemmer = stemmer):
    doc_stem = []
    for word in doc.split(' '):
        doc_stem.append(stemmer.stem(word.decode('utf-8')))
    return ' '.join(doc_stem)

def compare(doc_bow1, doc_bow2):
    top1 = {top[0] for top in doc_bow1}
    top2 = {top[0] for top in doc_bow2}
    top_inter = top1.intersection(top2)
    if not top_inter:
        return 0
    prod1 = filter(lambda x: x[0] in top_inter, doc_bow1)
    prod2 = filter(lambda x: x[0] in top_inter, doc_bow2)
    prod1.sort(key=lambda x : x[0])
    prod2.sort(key=lambda x : x[0])
    return sum([p1[1]*p2[1] for p1,p2 in zip(prod1, prod2)])

def kl(doc1, doc2, ntop):
    """
    Kullback-Leibler divergence.
    """
    d1 = np.repeat(0.000001, ntop)
    d2 = np.repeat(0.000001, ntop)
    for prop in doc1:
        d1[prop[0]] = prop[1]
    for prop in doc2:
        d2[prop[0]] = prop[1]
    return np.sum(d1*(np.log(d1/d2)))

def query_doc(doc, corpus, ntop):
    return [kl(doc, docc, ntop) for docc in corpus]


# In[5]:

sw = ['a', 'abord', 'afin']


# In[6]:

df = pd.read_csv(file_name)


# In[8]:

corpus = df.ldstatv.tolist()


# In[11]:

texts = []
t = 0
for text in corpus :
    text_cleaned = []
    try :
        text = clean(text).encode('utf-8')
        for element in text.split(' ') :
            if len(element) > 2 and element not in sw:
                text_cleaned.append(element.decode('utf-8'))
        texts.append(text_cleaned)
    except :
        t+=1
        texts.append([])


# In[12]:

texts[0]


# In[13]:

dictionary = corpora.Dictionary(texts)
DICT_FNAME = dict_file_name
dictionary.save_as_text(DICT_FNAME)


# In[14]:

corpus = [dictionary.doc2bow(text) for text in texts]


# In[15]:

corpora.BleiCorpus.serialize(corpora_file_name,corpus)


# In[16]:

bleiCorp = corpora.BleiCorpus(corpora_file_name)
id2word = corpora.Dictionary.load_from_text(DICT_FNAME)


# In[17]:

NB_TOPICS = 10
ALPHA = .0025
NB_RESULTS = 10
lda = models.ldamodel.LdaModel(corpus= bleiCorp,
                                   num_topics = NB_TOPICS,
                                   id2word = id2word,
                                   iterations = 300,
                                   chunksize=600,
                                   eval_every=1,
                                   alpha=ALPHA)


# In[18]:

new_docs = [id2word.doc2bow(text) for text in texts]
all_docs = [lda[new_doc] for new_doc in new_docs]


# In[19]:

lda.show_topics()


# In[20]:

all_docs[1]


# In[21]:

lda.show_topic(9)


# In[22]:

texts[1]


# In[ ]:
