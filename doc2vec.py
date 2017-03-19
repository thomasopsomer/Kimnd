#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from enron_graph_corpus import EnronGraphCorpus
from utils import load_dataset
import logging
import os
import pandas as pd


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class DocumentGenerator(object):
    """ """
    def __init__(self, egc, people=False):
        """ """
        self.egc = egc
        self.people = people
        df_train = self.egc.dataset.sample(frac=1).drop("recipients", axis=1)
        df_test = self.egc.df_test.sample(frac=1)
        self.df = pd.concat([df_train, df_test], axis=0)
    #
    def __iter__(self):
        """ """
        # shuffle dataset
        df = self.df.sample(frac=1)
        #
        for row in df.itertuples():
            mid = row.mid
            bow = [w for s in self.egc.doc[mid]["sents"] for w in s]
            if self.people:
                sender = row.sender
                recipients = list(row.recipients)
                yield TaggedDocument(bow, [sender] + recipients)
            else:
                yield TaggedDocument(bow, [mid])


#
path_to_data = os.path.join(os.getcwd(), 'data/')
ds_path = path_to_data + 'training_set.csv'
mail_path = path_to_data + 'training_info.csv'

ds_path_test = path_to_data + 'test_set.csv'
mail_path_test = path_to_data + 'test_info.csv'

df = load_dataset(ds_path, mail_path)
df_test = load_dataset(ds_path_test, mail_path_test, train=False)

egc = EnronGraphCorpus(df, df_test)
egc.load_doc("data/docall.pkl")

# doc generator
documents = DocumentGenerator(egc, people=False)

# ... train doc2vec model

model = Doc2Vec(size=300, min_count=3, iter=50, dm=1, workers=4)

# build vocab
model.build_vocab(documents)

# load pre-trained vectors
fname = "emb/GoogleNews-vectors-negative300.bin"
model.intersect_word2vec_format(fname, binary=True, lockf=1.0)

# train doc-vectors
model.train(documents)

model.save("emb/")

# directly learn word and doc vectors
model = Doc2Vec(documents, size=128, min_count=3, iter=30, dm=1, workers=4)




