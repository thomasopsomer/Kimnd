# coding: utf-8

import pandas as pd
import spacy
from gensim import corpora, models
from os import path

path_to_data = 'data'
training_info = pd.read_csv(path.join(path_to_data, 'training_info.csv'),
                                        sep=',', header=0)
training_info["body"] = training_info["body"].str.decode('utf-8')

nlp = spacy.load('en')

docs = nlp.pipe(training_info["body"], batch_size=1000, n_threads=4)

texts = []
for i, doc in enumerate(docs):
    texts.append([tok.lemma_ for tok in doc])
    if i%10 == 0:
        print i

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

# all_docs[1]
#
# lda.show_topic(9)
#
# texts[1]
