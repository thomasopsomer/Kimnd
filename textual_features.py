# coding: utf-8

import numpy as np
import pandas as pd
import utils
from gow import top_n_similarity, compute_idf
from gensim import corpora


def get_global_text_features(texts):
    """from list of token returns idf (dict), a gensim dictionary, and
    the average length"""
    id2word = corpora.Dictionary(texts)
    id2word.filter_extremes(no_below=4, no_above=0.2, keep_n=100000)
    idf, avg_len = compute_idf(texts, id2word)
    return idf, id2word, avg_len


def incoming_text_similarity(dataset, m, user, idf, id2word, avg_len, n):
    # Dataset containing all previous emails sent to person 'user'
    import pdb; pdb.set_trace()
    dataset_to_rec = dataset[dataset.recipient == user]
    # Measure similarity between m and all the messages received
    dataset_similar = top_n_similarity(n, m, dataset_to_rec, idf, id2word, avg_len)
    df_incoming = pd.DataFrame(columns=['user', 'sender', 'recipient', 'incoming text'])
    for c in dataset_similar['sender']:
        df_incoming = df_incoming.append(pd.DataFrame(
            [user, c, user, 1], columns=df_incoming.columns)
        )
    return df_incoming


def outgoing_text_similarity(dataset, m, user):
    # Dataset containing all previous emails sent by person 'user'
    dataset_from_rec = dataset[dataset.recipient == user]
    # Measure similarity between m and all the messages received
    ## METTRE TEXTS
    dataset_similar = top30_similarity(m, dataset_from_rec, texts)
    df_outgoing = pd.DataFrame(columns=['user', 'sender', 'recipient', 'outgoing text'])
    for c in dataset_similar['recipient']: ## TO CHANGE NAME, SEE WITH PIERRE
        df_outgoing = df_outgoing.append(pd.DataFrame(
            [user, user, c, 1], columns=df_outgoing.columns)
        )
    return df_outgoing

if __name__ == "__main__":

    dataset_path = "data/training_set.csv"
    mail_path = "data/training_info.csv"
    train_df = utils.load_dataset(dataset_path, mail_path, train=True, flat=True)

    train_df = utils.preprocess_bodies(train_df)
    texts = train_df["tokens"]
    # Compute idf
    idf, idf2word, avg_len = get_global_text_features(texts)
    # Message to compare
    message = texts[0]
    # Computing similarity between message and df_user_messages
    user = 'jshankm@enron.com'
    n = 5
    df_incoming = incoming_text_similarity(train_df, message, user, idf, idf2word, avg_len, n)
