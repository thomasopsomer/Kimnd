#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx
from utils import load_dataset, parallelize_fct, flatmap
from flat_dataset import parallelize_dataframe
from collections import Counter, defaultdict
from gensim.models.keyedvectors import KeyedVectors
# import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from math import ceil
from spacy_utils import get_custom_spacy, extract_nlp, preprocess_txt
import spacy
import cPickle as _pickle
from sklearn.preprocessing import normalize
import heapq


def sample_neg_recipient(row, G, emails, percent_neg=1):
    """
    recipient: list of recipient emails
    emails: set of all recipient emails in the dataset
    mail2id: dict mapping mail to their id
    percent: 0-1 float for percentage of negative samples
    """
    # neg emails among contact of the sender
    recipients = list(row.recipients)
    n_recipients = len(recipients)
    n_neg = int(percent_neg * n_recipients)
    # all neighbors of the sender
    contacts = set(G.neighbors(row.sender))
    #
    neg_contacts = list(contacts.difference(recipients))
    # We can't exceed the number of users
    n_neg_contact = min(len(neg_contacts), n_neg)
    if n_neg_contact == 0:
        neg_contacts = []
    else:
        neg_contacts = np.random.choice(neg_contacts, n_neg_contact)
    # neg emails among all emails except contact and recipient of the email
    if n_neg_contact < n_neg:
        n_neg_random = n_neg - n_neg_contact
        possible = list(emails.difference(list(neg_contacts) + list(recipients)))
        neg_random = np.random.choice(possible, n_neg_random)
        return list(neg_contacts) + list(neg_random)
    else:
        return list(neg_contacts)


def sample_neg_rec_df(df, G, emails, percent_neg=0.7):
    """ """
    df = df.copy()
    df["negs"] = df.apply(
        lambda x: sample_neg_recipient(
            x, G, emails, percent_neg), axis=1)
    # df["negs"] = df["negs"].map(lambda x: x[0])
    return df


def build_flat_dataset(df, G, emails, percent_neg=1,
                       dr=False, n_cores=4):
    """ """
    df = df.copy()[["sender", "recipients", "mid"]]
    if not dr:
        G = G.to_undirected()
    # emails = set(self.id2email)
    emails = set(emails)
    # create fake paires randomly
    if n_cores > 1:
        df_neg = parallelize_dataframe(
            df, sample_neg_rec_df, num_cores=n_cores,
            emails=emails, G=G, percent_neg=percent_neg)
    else:
        df_neg = sample_neg_rec_df(df, emails=emails, G=G,
                                   percent_neg=percent_neg)
    # flatten the recipients
    df_flat_rec = flatmap(df, "recipients", "recipient")
    df_flat_neg = flatmap(
        df_neg.drop("recipients", axis=1), "negs", "recipient")
    # add labels: 0 for fake recipient, 1 for others
    df_flat_rec["label"] = 1
    df_flat_neg["label"] = 0
    # concat neg and real recipient paires
    df_flat = pd.concat((df_flat_rec, df_flat_neg), axis=0)
    #
    return df_flat


def create_test_df(df, G, dr=False):
    """ """
    df = df.copy()
    if not dr:
        G = G.to_undirected()
    # df.sender = df.sender.map(lambda x: self.mail2id[x])
    # df.recipients = df.recipients.map(
    #     lambda x: [self.mail2id[m] for m in x])
    df["candidates"] = df.sender.map(lambda x: G.neighbors(x))
    df_flat = flatmap(df, "candidates", "candidate")
    return df_flat


def extract_bodies(docs):
    """ """
    res = []
    n = len(docs)
    for i, doc in enumerate(docs):
        sents, persons = extract_nlp(
            doc, bow=False, n_sentence=-1, index=False,
            people=True, s_max=-1)
        res.append({"sents": sents, "persons": persons})
        if i % int(n / 50) == 0:
                print "Progress: %.2f %%" % (100.0 * i / n)
    return res


class EnronGraphCorpus(object):
    """
    """
    def __init__(self, dataset, df_test=None):
        """
        dataset: dataframe builded with "utils.load_dataset"
        """
        self.dataset = dataset
        self.df_test = df_test
        # id2mail and mail2id mapping
        self.s_email = dataset.sender.unique()
        self.r_email = flatmap(dataset, "recipients", "recipient") \
            .recipient.unique()
        self.id2email = sorted(list(set(self.r_email).union(self.s_email)))
        self.mail2id = dict((m, k) for k, m in enumerate(self.id2email))
        self.num_doc = dataset.shape[0]
        self.doc = {}
        self.wcbow = {}

    def build_graph(self, output_path=None):
        """ """
        df = self.dataset
        # compoute frequency
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

        # save graph as edgelist txt
        if output_path is not None:
            # set node name to ids for use in node2vec later
            DG = nx.relabel_nodes(self.DG, self.mail2id)
            with open(output_path, "w") as f:
                for fr, to, data in nx.to_edgelist(DG):
                    f.write("%s %s %s\n" % (fr, to, data["weight"]))

    def load_n2v_vectors(self, vectors_path):
        """ """
        # load node vectors
        self.n2v = KeyedVectors.load_word2vec_format(vectors_path,
                                                     binary=False)
        # l2 normalize
        self.n2v.syn0 = normalize(self.n2v.syn0)
        for k, x in enumerate(self.n2v.index2word):
            mail = self.id2email[int(x)]
            voc = self.n2v.vocab[x]
            self.n2v.vocab[mail] = voc
            # self.n2v.vocab[int(x)] = voc
            del self.n2v.vocab[x]
            # self.n2v.index2word[k] = int(x)
            self.n2v.index2word[k] = mail
        #
        return

    def load_d2v_vectors(self, d2v_model_path):
        """ """
        # load doc2vec vectors
        from gensim.models.doc2vec import Doc2Vec
        model = Doc2Vec.load(d2v_model_path)
        model.docvecs.doctag_syn0 = normalize(model.docvecs.doctag_syn0)
        self.d2v = model.docvecs
        #
        return

    def fit_idf(self, min_df=0):
        """ """
        doc_freq = Counter()
        for mid, doc in self.doc.iteritems():
            bow = [x for s in doc["sents"] for x in s]
            unique_tokens = [x for x in list(set(bow)) if x]
            doc_freq.update(unique_tokens)
        self.idf = dict((k, v) for k, v in doc_freq.iteritems()
                        if v > min_df)
        return

    def parse_bodies(self, n_sentence=-1, batch_size=100, n_threads=4):
        """ """
        # load spacy
        if not hasattr(self, "nlp"):
            print("Initiating spacy engine...")
            self.nlp = get_custom_spacy(entity=True, parser=True)
        #
        mids = self.dataset.mid.values
        bodies = self.dataset.body.values

        if self.df_test is not None:
            mids = np.hstack([mids, self.df_test.mid.values])
            bodies = np.hstack([bodies, self.df_test.body.values])

        # preprocess bodies
        print("Preprocessing raw text...")
        bodies = parallelize_fct(
            func=preprocess_txt, arg_list=bodies, num_cores=4)
        print("Parsing and extraction using spacy...")

        # extract from spacy parsed doc
        res = extract_bodies(self.nlp.pipe(bodies, batch_size=batch_size,
                             n_threads=n_threads))
        for i, d in enumerate(res):
            self.doc[mids[i]] = d

        # extract from spacy parsed doc
        # n = len(bodies)
        # for i, doc in enumerate(self.nlp.pipe(bodies,
        #                                       batch_size=100,
        #                                       n_threads=4)):
        #     sents, persons = extract_nlp(doc, bow=False, n_sentence=n_sentence)
        #     self.doc[mids[i]] = {"sents": sents, "persons": persons}
        #     if i % int(n / 50) == 0:
        #         print "Progress: %.2f %%" % (100.0 * i / n)
        return

    def load_doc(self, doc_path):
        """ """
        with open(doc_path, 'r') as f:
            self.doc = _pickle.load(f)
        return True

    def make_wcbow(self, w="uniform", tf_mode="log", k=15):
        """
        for now use spacy vectors
        """
        # spacy to load vectors :)
        if not hasattr(self, "nlp"):
            self.nlp = get_custom_spacy()
        #
        for mid, doc in self.doc.iteritems():
            bow = [x for s in doc["sents"] for x in s]
            if not bow:
                self.wcbow[mid] = np.zeros(300, dtype=np.float32)
            else:
                if w == "uniform":
                    doc = spacy.tokens.doc.Doc(self.nlp.vocab, bow)
                    self.wcbow[mid] = doc.vector
                elif w == "tfidf":
                    tf = Counter(bow)
                    w_tfidf = []
                    for word in bow:
                        if word in self.idf:
                            idf = compute_idf(self.num_doc, self.idf[word])
                            tfidf = compute_tfidf(tf[word], idf, tf_mode=tf_mode)
                            w_tfidf.append((word, tfidf))
                    # filter k keywords
                    if k > 0:
                        w_tfidf = heapq.nlargest(k, w_tfidf, key=lambda x: x[1])
                    #
                    if w_tfidf:
                        words, weights = zip(*w_tfidf)
                        doc = spacy.tokens.doc.Doc(self.nlp.vocab, words)
                        self.wcbow[mid] = np.average([tok.vector for tok in doc],
                                                     weights=weights, axis=0)
                    else:
                        self.wcbow[mid] = np.zeros(300, dtype=np.float32)
        return

    def get_wcbow(self):
        return self.wcbow

    def build_people_vectors(self, d2v=False):
        """ """
        # sender (avg vector of all message send)
        self.sender_rep = {}
        se = self.dataset.groupby("sender")["mid"].apply(list).to_dict()
        for k, v in se.iteritems():
            if d2v:
                vecs = [self.d2v[mid] for mid in v]
            else:
                vecs = [self.wcbow[mid] for mid in v]
            self.sender_rep[k] = np.average(vecs, axis=0)

        # recipient (avg vector of all message received)
        self.recipient_rep = {}
        d = flatmap(self.dataset, "recipients", "recipient") \
            .groupby("recipient")["mid"] \
            .apply(list) \
            .to_dict()
        for k, v in d.iteritems():
            if d2v:
                vecs = [self.d2v[mid] for mid in v]
            else:
                vecs = [self.wcbow[mid] for mid in v]
            self.recipient_rep[k] = np.average(vecs, axis=0)
        return

    def build_recipient_sender_vectors(self, d2v=False):
        """ """
        # recipient | sender rep
        df = self.dataset
        # flatten dataset
        df_flat = flatmap(df[["sender", "mid", "recipients"]],
                          "recipients", "recipient")
        # groupby recipient then sender, collect list on mid
        rep = df_flat.groupby(["recipient", "sender"])["mid"].apply(list) \
            .to_dict()
        #
        for tupl in rep.keys():
            if d2v:
                rep[tupl] = np.average([self.d2v[m] for m in rep[tupl]],
                                       axis=0)
            else:
                rep[tupl] = np.average([self.wcbow[m] for m in rep[tupl]],
                                       axis=0)
        self.recipient_sender_rep = rep
        return

    def get_sender_rep(self):
        """ """
        return self.sender_rep

    def get_recipient_rep(self):
        """ """
        return self.recipient_rep

    def save(self, output_path):
        """ """
        with open(output_path, "w") as fout:
            _pickle.dump(self, fout)

    @classmethod
    def load(self, input_path):
        """ """
        with open(input_path, "r") as f:
            obj = _pickle.load(f)
        return obj


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


def compute_tfidf(tf, idf, tf_mode="raw", **kwarg):
    """ """
    if tf_mode == "log":
        tf = 1.0 + np.log(tf)
    elif tf_mode == "loglog":
        tf = 1.0 + np.log(1.0 + np.log(tf))
    elif tf_mode == "sqrt":
        tf = np.sqrt(tf)
    # elif tf_mode == "pol":
    #     tf = (1.0 + np.log(tf)) / (1 - b + b * len_doc / avg)
    return tf * idf


# if __name__ == '__main__':

#     # load node vectors
#     # n2v = KeyedVectors.load_word2vec_format('emb/enron_p_1_q_03_u.emb',
#     #                                         binary=False)
#     # for k, x in enumerate(n2v.index2word):
#     #     mail = id2email[int(x)]
#     #     voc = n2v.vocab[x]
#     #     n2v.vocab[mail] = voc
#     #     del n2v.vocab[x]
#     #     n2v.index2word[k] = mail

#     # n2v.similarity("karen.buckley@enron.com", "andrew.h.lewis@enron.com")

#     # for x, c in freq["karen.buckley@enron.com"].iteritems():
#     #     print x, c
#     #     print n2v.similarity("karen.buckley@enron.com", x)

#     # r = n2v.most_similar("  ", topn=None)

#     # plt.hist(r, bins=100)
#     # plt.show()
#     import os
#     from utils import load_dataset
#     from enron_graph_corpus import EnronGraphCorpus
#     #
#     path_to_data = os.path.join(os.getcwd(), 'data/')
#     # train data
#     ds_path = path_to_data + 'training_set.csv'
#     mail_path = path_to_data + 'training_info.csv'
#     # test data
#     ds_path_test = path_to_data + 'test_set.csv'
#     mail_path_test = path_to_data + 'test_info.csv'

#     # clean a bit
#     df = load_dataset(ds_path, mail_path)
#     df_test = load_dataset(ds_path_test, mail_path_test, train=False)

#     egc = EnronGraphCorpus(df, df_test=df_test)

#     egc.build_graph("data/enron.edgelist")
#     egc.DG
#     egc.load_n2v_vectors("emb/enron_p_1_q_05_d.emb")
#     egc.n2v.most_similar("karen.buckley@enron.com")

#     # parse bodies
#     egc.parse_bodies()
#     egc.load_doc("data/doc.pkl")
#     egc.fit_idf(min_df=4)
#     egc.make_wcbow(w="tfidf", k=-1)
#     egc.build_people_vectors()
#     egc.build_recipient_sender_vectors()

#     # egc.make_bow()
#     egc.load_doc("data/doc.pkl")
#     egc.fit_idf(min_df=4)
#     egc.make_wcbow(w="tfidf")

#     egc.build_people_vectors()


#     import cPickle

#     with open("data/docall.pkl", "w") as f:
#         obj = egc.doc
#         cPickle.dump(obj, f)

#     with open("data/recipient_sender_rep.pkl", "w") as f:
#         obj = egc.recipient_sender_rep
#         cPickle.dump(obj, f)

#     with open("data/msg_rep.pkl", "w") as f:
#         obj = egc.wcbow
#         cPickle.dump(obj, f)











