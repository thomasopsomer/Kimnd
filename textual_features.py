# coding: utf-8

import numpy as np
import pandas as pd
import utils
from gow import top_n_similarity, compute_idf
from gensim import corpora
import cPickle as pkl


def get_global_text_features(texts):
    """from list of token returns idf (dict), a gensim dictionary, and
    the average length"""
    id2word = corpora.Dictionary(texts)
    id2word.filter_extremes(no_below=10, no_above=0.2, keep_n=100000)
    idf, avg_len = compute_idf(texts, id2word)
    return idf, id2word, avg_len


def incoming_text_similarity(dataset, mid, user, idf, id2word, avg_len, n):
    # Dataset containing all previous emails sent to person 'user'
    dataset_to_rec = dataset[dataset['recipients'].map(lambda x: user in x)]
    # Measure similarity between the message of id 'mid' and all the messages received
    message = dataset[dataset['mid'] == mid]['tokens'].values[0]
    dataset_similar = top_n_similarity(n, message, dataset_to_rec, idf, id2word, avg_len)
    df_incoming = pd.DataFrame(columns=['mid', 'user', 'contact', 'incoming text'], )
    list_sender = np.unique(dataset['sender'].tolist())
    for c in list_sender:
        if c in dataset_similar['sender']:
            df_incoming = df_incoming.append(pd.DataFrame(
                [[mid, user, c, 1]], columns=df_incoming.columns)
            )
        else:
            df_incoming = df_incoming.append(pd.DataFrame(
                [[mid, user, c, -1]], columns=df_incoming.columns)
            )
    pdb.set_trace()
    return df_incoming


def outgoing_text_similarity(dataset, mid, user, idf, id2word, avg_len, n):
    # Dataset containing all previous emails sent by person 'user'
    dataset_from_rec = dataset[dataset.sender == user]
    # Measure similarity between the message of id 'mid' and all the messages sent
    message = dataset[dataset['mid'] == mid]['tokens'].values[0]
    dataset_similar = top_n_similarity(n, message, dataset_from_rec, idf, id2word, avg_len)
    df_outgoing = pd.DataFrame(columns=['mid', 'user', 'contact', 'outgoing text'])
    dataset_flat = utils.flatmap(dataset, "recipients", "recipient", np.string_)
    list_recipients = np.unique(dataset_flat['recipient'].tolist())
    list_recipients_similar = np.unique(sum(dataset_similar['recipients'].tolist(), []))
    for c in list_recipients:
        if c in list_recipients_similar:
            df_outgoing = df_outgoing.append(pd.DataFrame(
                [[mid, user, c, 1]], columns=df_outgoing.columns)
            )
        else:
            df_outgoing = df_outgoing.append(pd.DataFrame(
                [[mid, user, c, -1]], columns=df_outgoing.columns)
            )
    pdb.set_trace()
    return df_outgoing


if __name__ == "__main__":
    dataset_path = "data/training_set.csv"
    mail_path = "data/training_info.csv"
    train_df = utils.load_dataset(dataset_path, mail_path, train=True)
    train_df = utils.preprocess_bodies(train_df)
    texts = train_df["tokens"]
    # Compute idf
    #idf, id2word, avg_len = get_global_text_features(texts)
    idf = pkl.load(open('idf'))
    id2word = pkl.load(open('id2word'))
    avg_len = 66
    import pdb; pdb.set_trace()
    # Message to compare
    mid = 158713
    # Computing similarity between message and df_user_messages
    user = 'jarnold@enron.com'
    n = 5
    df_incoming = incoming_text_similarity(train_df, mid, user, idf, id2word, avg_len, n)
    user = 'karen.buckley@enron.com'
    df_outgoing = outgoing_text_similarity(train_df, mid, user, idf, id2word, avg_len, n)
    print('Done.')
