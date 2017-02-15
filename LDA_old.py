# coding: utf-8

import pandas as pd
import re
import numpy as np
from collections import Counter
from ast import literal_eval
from gensim import corpora, models




sw = ['a', 'abord', 'afin']

df = pd.read_csv(file_name)

corpus = df.ldstatv.tolist()

texts = []
t = 0
for text in corpus:
    text_cleaned = []
    try:
        text = clean(text).encode('utf-8')
        for element in text.split(' '):
            if len(element) > 2 and element not in sw:
                text_cleaned.append(element.decode('utf-8'))
        texts.append(text_cleaned)
    except:
        t += 1
        texts.append([])

dictionary = corpora.Dictionary(texts)
DICT_FNAME = dict_file_name
dictionary.save_as_text(DICT_FNAME)

corpus = [dictionary.doc2bow(text) for text in texts]

corpora.BleiCorpus.serialize(corpora_file_name, corpus)

bleiCorp = corpora.BleiCorpus(corpora_file_name)
id2word = corpora.Dictionary.load_from_text(DICT_FNAME)

NB_TOPICS = 10
ALPHA = .0025
NB_RESULTS = 10
lda = models.ldamodel.LdaModel(corpus=bleiCorp,
                               num_topics=NB_TOPICS,
                               id2word=id2word,
                               iterations=300,
                               chunksize=600,
                               eval_every=1,
                               alpha=ALPHA)


new_docs = [id2word.doc2bow(text) for text in texts]
all_docs = [lda[new_doc] for new_doc in new_docs]

lda.show_topics()

all_docs[1]

lda.show_topic(9)

texts[1]
