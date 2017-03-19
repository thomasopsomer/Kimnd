# coding: utf-8

import numpy as np
import pandas as pd
from scipy.sparse import vstack, csr_matrix
import utils
from gow import top_n_similarity, compute_idf
from gensim import corpora
from sklearn.preprocessing import normalize


def get_global_text_features(texts):
    """
    From a list of tokens, returns idf (dict), a gensim dictionary, and
    the average length
    """
    id2word = corpora.Dictionary(texts)
    id2word.filter_extremes(no_below=10, no_above=0.2, keep_n=100000)
    idf, avg_len = compute_idf(texts, id2word)
    return idf, id2word, avg_len


def text_similarity_new(df_flat, dico_twidf, dico_average_twidf):
    """
    Computes incoming or outgoing textual features similarity: cosine similarity between a list of
    messages and all the messages received or sent by users
    :param df_flat: original flatten dataframe
    :param dico_twidf: dictionary where keys: messages id and items: their textual representation (eg: twidf)
    :param dico_average_twidf: dictionary where keys: users and items: average textual representation of
    messages sent or received by the user
    :return:
    """
    len_twidf = dico_twidf.values()[0].shape[1]
    not_found = csr_matrix(np.zeros(len_twidf))
    mids = vstack([dico_twidf[x] for x in df_flat.mid])
    mids = normalize(mids)
    averages = vstack([dico_average_twidf.get((row[1].sender, row[1].recipient), not_found) for row in df_flat.iterrows()])
    averages = normalize(averages)
    return np.sum(mids.multiply(averages), axis=1)


def incoming_text_similarity(dataset, mid, user, twidf_df, n):
    """
    Computing incoming textual similarity
    TOO SLOW: was not used!
    """
    # Dataset containing all previous emails sent to person 'user'
    dataset_to_rec = dataset[dataset['recipients'].map(lambda x: user in x)]
    # Measure similarity between the message of id 'mid' and all the messages received
    dataset_similar = top_n_similarity(n, mid, dataset_to_rec, twidf_df)
    df_incoming = pd.DataFrame(columns=['mid', 'user', 'contact', 'incoming_text'])
    list_sender = np.unique(dataset['sender'].tolist())
    df_incoming['contact'] = pd.Series(list_sender)
    df_incoming['mid'] = mid
    df_incoming['user'] = user
    df_incoming['incoming_text'] = pd.Series([1 if c in dataset_similar['sender'] else -1 for c in list_sender])
    return df_incoming


def outgoing_text_similarity(dataset, mid, user, twidf_df, n):
    """
    Computing outgoing textual similarity
    TOO SLOW: was not used!
    """
    # Dataset containing all previous emails sent by person 'user'
    dataset_from_rec = dataset[dataset.sender == user]
    # Measure similarity between the message of id 'mid' and all the messages sent
    dataset_similar = top_n_similarity(n, mid, dataset_from_rec, twidf_df)
    df_outgoing = pd.DataFrame(columns=['mid', 'user', 'contact', 'outgoing_text'])
    dataset_flat = utils.flatmap(dataset, "recipients", "recipient", np.string_)
    list_recipients = np.unique(dataset_flat['recipient'].tolist())
    list_recipients_similar = np.unique(sum(dataset_similar['recipients'].tolist(), []))
    df_outgoing['contact'] = pd.Series(list_recipients)
    df_outgoing['mid'] = mid
    df_outgoing['user'] = user
    df_outgoing['outgoing_text'] = pd.Series([1 if c in list_recipients_similar else -1 for c in list_recipients])
    return df_outgoing
