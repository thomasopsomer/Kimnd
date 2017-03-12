# coding: utf-8
from os import path
from ast import literal_eval
import argparse
import re

import pandas as pd
import spacy
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim

from utils import bow_mail_body
from spacy_utils import get_custom_spacy


def flatten_topics(row, nb_topics):
    dict_topics = {i: 0 for i in range(nb_topics)}
    for (topic, score) in row[0]:
        dict_topics[topic] = score
    return pd.Series(dict_topics)


def compute_lda(data_path, load_path=None, output_path=None, tfidf=True,
                nb_topics=4, alpha=.0025):
    print "Loading data"
    training_info = pd.read_csv(path.join(data_path, "training_info.csv"),
                                sep=',', header=0).set_index("mid")
    test_info = pd.read_csv(path.join(data_path, "test_info.csv"),
                            sep=',', header=0).set_index("mid")
    texts = pd.concat([training_info["body"], test_info["body"]])
    mids = list(texts.index)
    if output_path is None:
        output_path = "LDA_data"
    if load_path is None:
        texts = texts.str.decode('utf-8')
        print "Loading Spacy"
        nlp = get_custom_spacy()

        print "Preprocessing mails"
        texts = texts.apply(bow_mail_body, args=(nlp,))

        texts = list(texts)

        print "Creating id2word and corpus"
        id2word = corpora.Dictionary(texts)
        id2word.filter_extremes(no_below=4, no_above=0.2, keep_n=100000)
        id2word.save_as_text(output_path + "_dic")

        corpus = [id2word.doc2bow(text) for text in texts]
        if tfidf:
            tfidf_corpus = models.TfidfModel(corpus)
            corpus = tfidf_corpus[corpus]
        corpora.BleiCorpus.serialize(output_path + "_corp", corpus)
    else:
        print "Loading id2word and corpus"
        corpus = corpora.BleiCorpus(load_path + "_corp")
        id2word = corpora.Dictionary.load_from_text(load_path + "_dic")

    print "Applying LDA"
    #NB_RESULTS = 10
    lda = models.ldamulticore.LdaMulticore(workers=3,
                                       corpus=corpus,
                                       num_topics=nb_topics,
                                       id2word=id2word,
                                       iterations=300,
                                       chunksize=600,
                                       eval_every=1,
                                       alpha=alpha)

    print lda.show_topics()
    lda.save(output_path+ "_lda")

    print "Saving the csv with topics for each mail"
    all_docs = [lda[doc] for doc in corpus]
    all_docs = pd.DataFrame({"mids": mids,
                            "lda_res": all_docs}).set_index("mids")
    all_docs = all_docs.apply(flatten_topics, args=(nb_topics,), axis=1)
    all_docs.to_csv("LDA_results.csv", sep=",", index=True)


def vis_lda(load_path, output_html=None):
    print "Visualizing the LDA"
    if load_path is None:
        load_path = "LDA_data"
    if output_html is None:
        output_html = "LDA_vis.html"
    corpus = corpora.BleiCorpus(load_path + "_corp")
    id2word = corpora.Dictionary.load_from_text(load_path + "_dic")
    lda = models.ldamodel.LdaModel.load(load_path+ "_lda")
    vis = pyLDAvis.gensim.prepare(lda, corpus, id2word)
    pyLDAvis.save_html(vis, output_html)


def parse_args():
    parser = argparse.ArgumentParser("Extract topics with LDA on a "
                                     "text database")

    parser.add_argument("data_path",
                        help="path to the folder of csv files containing the"
                        " mails")
    parser.add_argument("-l", "--load",
                        help="path to load the dictionnary and corpus for LDA")
    parser.add_argument("-o", "--output",
                        help="path to save the dictionnary and corpus for the"
                        " LDA")
    parser.add_argument("-ht", "--html",
                        help="path to save the html file for visualiazin the"
                        " LDA")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    compute_lda(args.data_path, load_path=args.load, output_path=args.output)
    vis_lda(args.output, output_html=args.html)
