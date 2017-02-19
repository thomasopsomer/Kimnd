# coding: utf-8
from os import path
import argparse

import pandas as pd
import spacy
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim

from utils import preprocess_mail_body


def compute_lda(data_path, load_path=None, output_path=None, tfidf=False):
    if output_path is None:
        output_path = "LDA_data"
    if load_path is None:
        print "Loading data"
        training_info = pd.read_csv(data_path, sep=',', header=0)
        training_info["body"] = training_info["body"].str.decode('utf-8')
        print "Loading Spacy"
        nlp = spacy.load('en', parser=False)

        # docs = nlp.pipe(training_info.iloc[:1000]["body"], batch_size=1000,
        #                 n_threads=4)
        print "Preprocessing mails"
        texts = training_info["body"].apply(preprocess_mail_body, args=(nlp,))
        texts = list(texts)
        # for i, doc in enumerate(training_info["body"]):
        #     texts.append(preprocess_mail_body(doc, nlp))
        #     if i % 1000 == 0:
        #         print "{:d} document processed".format(i)
        print "Creating id2word and corpus"
        id2word = corpora.Dictionary(texts)

        id2word.save_as_text(output_path + "_dic")

        corpus = [id2word.doc2bow(text) for text in texts]
        if tfidf:
            corpus = models.TfidfModel(corpus)

        corpora.BleiCorpus.serialize(output_path + "_corp", corpus)
    else:
        print "Loading id2word and corpus"
        corpus = corpora.BleiCorpus(load_path + "_corp")
        id2word = corpora.Dictionary.load_from_text(load_path + "_dic")

    print "Applying LDA"
    NB_TOPICS = 10
    ALPHA = .0025
    NB_RESULTS = 10
    lda = models.ldamulticore.LdaMulticore(workers=3,
                                       corpus=corpus,
                                       num_topics=NB_TOPICS,
                                       id2word=id2word,
                                       iterations=300,
                                       chunksize=600,
                                       eval_every=1,
                                       alpha=ALPHA)

    #all_docs = [lda[doc] for doc in corpus]

    print lda.show_topics()

    lda.save(output_path+ "_lda")


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
                        help="path to the csv file containing the mails")
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
    vis_lda(arg.output, output_html=args.html)
