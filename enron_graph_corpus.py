#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
from utils import load_dataset, flatmap
from collections import Counter, defaultdict
from gensim.models.keyedvectors import KeyedVectors
# import matplotlib.pyplot as plt
import os
import numpy as np
import multiprocessing as mp
import pandas as pd
from functools import partial
from math import ceil
from spacy_utils import get_custom_spacy, bow_mail_body, sent_mail_body
import spacy
import cPickle as Pickle


def sample_neg_recipient(recipients, emails, percent=0.3):
    """
    recipient: list of recipient emails
    emails: set of all emails
    mail2id: dict mapping mail to their id
    percent: 0-1 float for percentage of negative samples
    """
    n_rec = len(recipients)
    n_neg = int(ceil(percent * n_rec))
    # sample randomly n_neg fake recipient among the negative emails
    neg_emails = list(emails.difference(recipients))
    neg_samples = np.random.choice(neg_emails, n_neg)
    return neg_samples


def sample_neg_rec_df(df, emails, percent=0.3):
    """ """
    df["negs"] = df.recipients.map(
        lambda x: sample_neg_recipient(
            x, emails, percent=percent))
    return df


def parallelize_dataframe(df, func, num_cores, **kwargs):
    """ """
    df_split = np.array_split(df, num_cores)
    pool = mp.Pool(num_cores)
    partial_f = partial(func, **kwargs)
    try:
        print 'starting the pool map'
        df = pd.concat(pool.map(partial_f, df_split))
        pool.close()
        print 'pool map complete'
    except KeyboardInterrupt:
        print 'got ^C while pool mapping, terminating the pool'
        pool.terminate()
        print 'pool is terminated'
    except Exception, e:
        print 'got exception: %r, terminating the pool' % (e,)
        pool.terminate()
        print 'pool is terminated'
    finally:
        print 'joining pool processes'
        pool.join()
        print 'join complete'
    print 'the end'
    return df


class EnronGraphCorpus(object):
    """ """
    def __init__(self, dataset):
        """
        dataset: dataframe builded with "utils.load_dataset"
        """
        self.dataset = dataset
        # build graph
        self.build_graph()

    def build_graph(self, output_path=None):
        """ """
        df = self.dataset
        # all email and mapping to ids
        s_email = df.sender.unique()
        r_email = flatmap(df, "recipients", "recipient").recipient.unique()
        self.id2email = list(set(r_email).union(s_email))
        self.mail2id = dict((m, k) for k, m in enumerate(self.id2email))

        freq = defaultdict(Counter)
        for row in df.itertuples():
            sender = row.sender
            freq[sender].update(row.recipients)

        # create graph
        self.DG = nx.DiGraph()
        # fill the DiGraph
        for sender in freq.keys():
            for recipients, f in freq[sender].iteritems():
                self.DG.add_edge(sender, recipients, weight=f)
        # set node name to ids for use in node2vec later
        self.DG = nx.relabel_nodes(self.DG, self.mail2id)

        # save graph as edgelist txt
        if output_path is not None:
            with open(path_to_edgelist, "w") as f:
                for fr, to, data in nx.to_edgelist(DG):
                    f.write("%s %s %s\n" % (fr, to, data["weight"]))

    def build_flat_dataset(self, fake_percent=0.4):
        """ """
        emails = set(self.id2email)
        # create fake paires randomly
        df_neg = parallelize_dataframe(
            df, sample_neg_rec_df, num_cores=4, emails=emails,
            percent=fake_percent)

        # flatten the recipients
        df_flat_rec = flatmap(df, "recipients", "recipient")
        df_flat_neg = flatmap(
            df_neg.drop("recipients", axis=1), "negs", "recipient")
        # add labels: 0 for fake recipient, 1 for others
        df_flat_rec["label"] = 1
        df_flat_neg["label"] = 0
        # concat neg and real recipient paires
        df_flat = pd.concat((df_flat_rec, df_flat_neg), axis=0)
        # alias mail with ids
        df_flat.sender = df_flat.sender.map(lambda x: mail2id[x])
        df_flat.recipient = df_flat.recipient.map(lambda x: mail2id[x])
        #
        return df_flat

    def load_n2v_vectors(self, vectors_path):
        """ """
        # load node vectors
        self.n2v = KeyedVectors.load_word2vec_format(vectors_path,
                                                     binary=False)
        #
        for k, x in enumerate(self.n2v.index2word):
            mail = self.id2email[int(x)]
            voc = self.n2v.vocab[x]
            self.n2v.vocab[mail] = voc
            del self.n2v.vocab[x]
            self.n2v.index2word[k] = mail
        #
        return

    def fit_idf(self, min_df=0):
        """ """
        doc_freq = Counter()
        for mid, bow in self.bow.iteritems():
            unique_tokens = [x for x in list(set(bow)) if x]
            doc_freq.update(unique_tokens)
        self.idf = dict((k, v) for k, v in doc_freq.iteritems()
                        if v > min_df)
        return

    def make_bow(self):
        """ """
        self.nlp = get_custom_spacy()
        self.bow = {}
        n = self.dataset.shape[0]
        for i, row in enumerate(self.dataset.itertuples()):
            mid = row.mid
            bow = bow_mail_body(row.body, self.nlp)
            self.bow[mid] = bow
            if i % int(n / 50) == 0:
                print "Progress: %.2f %%" % (100.0 * i / n)
        return True

    def get_bow(self):
        return self.bow

    def load_bow(self, bow_path):
        """ """
        with open(bow_path, 'r') as f:
            self.bow = Pickle.load(f)
        return True

    def make_wcbow(self, w="uniform", mode="log"):
        """
        for now use spacy vectors
        """
        self.wcbow = {}
        for mid, bow in self.bow.iteritems():
            if not bow:
                self.wcbow[mid] = np.zeros(300, dtype=np.float32)
            else:
                if w == "uniform":
                    self.wcbow[mid] = doc.vector
                elif w == "tfidf":
                    tf = Counter(bow)
                    words, tfidf = []
                    for word in bow:
                        if word in self.idf:
                            words.append(word)
                            tfidf.append(compute_tfidf(tf[word], self.idf[word],
                                         mode=mode))
                    doc = spacy.tokens.doc.Doc(self.nlp.vocab, words)
                    self.wcbow[mid] = np.mean([tok.vector for tok in doc],
                                              weights=tfidf, axis=0)
        return

    def get_wcbow(self):
        return self.wcbow

    def build_people_vectors(self):
        """ """
        # sender (avg vector of all message send)
        self.sender_rep = {}
        se = self.dataset.groupby("sender")["mid"].apply(list).to_dict()
        for k, v in se.iteritems():
            vecs = [self.wcbow[mid] for mid in v]
            self.sender_rep[k] = np.mean(vecs)
        return

        # recipient (avg vector of all message received)
        self.recipient_rep = {}
        d = flatmap(self.dataset, "recipients", "recipient") \
            .groupby("recipient")["mid"] \
            .apply(list) \
            .to_dict()
        for k, v in d.iteritems():
            vecs = [self.wcbow[mid] for mid in v]
            self.recipient_rep[k] = np.mean(vecs)
        return

    def get_sender_rep(self):
        """ """
        return self.sender_rep

    def get_recipient_rep(self):
        """ """
        return self.recipient_rep


def compute_idf(D, DF, mode="tfidf"):
    """ """
    if mode.lower() == "tfidf":
        idf = np.log(float(D + 1) / float(DF + 1))
        return idf
    elif mode.lower() == "bm25":
        idf = np.log(1.0 * (D - DF + 0.5) / (DF + 0.5))
        return idf
    else:
        raise ValueError("""Mode needs to be tfidf of bm25""")


def compute_tfidf(tf, idf, len_doc, avg, b, tf_mode="raw",
                  **kwarg):
    """ """
    if tf_mode == "log":
        tf = 1.0 + np.log(tf)
    elif tf_mode == "loglog":
        tf = 1.0 + np.log(1.0 + np.log(tf))
    elif tf_mode == "sqrt":
        tf = np.sqrt(tf)
    elif tf_mode == "pol":
        tf = (1.0 + np.log(tf)) / (1 - b + b * len_doc / avg)
    return tf * idf


if __name__ == '__main__':

    # load node vectors
    # n2v = KeyedVectors.load_word2vec_format('emb/enron_p_1_q_03_u.emb',
    #                                         binary=False)
    # for k, x in enumerate(n2v.index2word):
    #     mail = id2email[int(x)]
    #     voc = n2v.vocab[x]
    #     n2v.vocab[mail] = voc
    #     del n2v.vocab[x]
    #     n2v.index2word[k] = mail

    # n2v.similarity("karen.buckley@enron.com", "andrew.h.lewis@enron.com")

    # for x, c in freq["karen.buckley@enron.com"].iteritems():
    #     print x, c
    #     print n2v.similarity("karen.buckley@enron.com", x)

    # r = n2v.most_similar("  ", topn=None)

    # plt.hist(r, bins=100)
    # plt.show()

    import os
    from utils import load_dataset
    from enron_graph_corpus import EnronGraphCorpus
    #
    path_to_data = os.path.join(os.getcwd(), 'data/')
    ds_path = path_to_data + 'training_set.csv'
    mail_path = path_to_data + 'training_info.csv'

    # clean a bit
    df = load_dataset(ds_path, mail_path)

    egc = EnronGraphCorpus(df)

    egc.build_graph()
    egc.DG
    egc.load_n2v_vectors("emb/enron_p_1_q_03_u.emb")
    egc.n2v.most_similar("karen.buckley@enron.com")

    egc.make_bow()
    egc.fit_idf(min_df=4)
    egc.make_wcbow(w="uniform")









