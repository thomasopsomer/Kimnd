# coding: utf-8

import pandas as pd
import spacy
from gensim import corpora, models
# import pyldavis
from os import path
import argparse


def compute_lda(data_path, load_path=None, output_path=None, tfidf=False):
    if output_path is None:
        output_path = "LDA_data"
    if load_path is None:
        training_info = pd.read_csv(data_path, sep=',', header=0)
        training_info["body"] = training_info["body"].str.decode('utf-8')

        nlp = spacy.load('en')

        docs = nlp.pipe(training_info.iloc[:1000]["body"], batch_size=1000,
                        n_threads=4)

        texts = []
        for i, doc in enumerate(docs):
            texts.append([tok.lemma_ for tok in doc])
            if i % 10 == 0:
                print i

        id2word = corpora.Dictionary(texts)

        id2word.save_as_text(output_path + "dic.txt")

        corpus = [id2word.doc2bow(text) for text in texts]
        if tfidf:
            corpus = models.TfidfModel(corpus)

        corpora.BleiCorpus.serialize(output_path, corpus)
    else:
        corpus = corpora.BleiCorpus(load_path)
        id2word = corpora.Dictionary.load_from_text(load_path + "dic.txt")

    NB_TOPICS = 10
    ALPHA = .0025
    NB_RESULTS = 10
    lda = models.ldamodel.LdaModel(corpus=corpus,
                                   num_topics=NB_TOPICS,
                                   id2word=id2word,
                                   iterations=300,
                                   chunksize=600,
                                   eval_every=1,
                                   alpha=ALPHA)

    new_docs = corpus
    all_docs = [lda[new_doc] for new_doc in new_docs]

    print lda.show_topics()

    # all_docs[1]
    # lda.show_topic(9)
    # texts[1]


def parse_args():
    parser = argparse.ArgumentParser("Extract topics with LDA on a "
                                     "text database")

    parser.add_argument("data_path",
                        help="path to the csv file containing the mails")
    parser.add_argument("-l", "--load",
                        help="path to load the dictionnary and corpus for LDA")
    parser.add_argument("-o", "--output",
                        help="path to save the dictionnary and corpus for the"
                        " LDA")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    compute_lda(args.data_path, load_path=args.load, output_path=args.output)
