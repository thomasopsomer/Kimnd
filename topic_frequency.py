import random
import operator
import os

import pandas as pd
import numpy as np  # FIXME
from collections import Counter
from gensim import corpora, models
import spacy

from utils import load_dataset, bow_mail_body
from average_precision import mapk

from spacy_utils import get_custom_spacy
import pdb
nlp = get_custom_spacy()

path_to_data = "data"
lda_path = "LDA/LDA_data_lda"
id2word = corpora.Dictionary.load_from_text("LDA/LDA_data_dic")



###################
# Dataset loading #
###################
dataset_path = os.path.join(path_to_data, 'training_set.csv')
mail_path = os.path.join(path_to_data, 'training_info.csv')

dataset = load_dataset(dataset_path, mail_path)

testset_path = os.path.join(path_to_data, 'test_set.csv')
testmail_path = os.path.join(path_to_data, 'test_info.csv')

testset = load_dataset(testset_path, testmail_path, train=False)

print "data loaded"
# load lda
lda = models.ldamodel.LdaModel.load(lda_path)
n_topics = lda.num_topics


def get_lda_score(text, lda, nlp, dic):
    topic_score = np.zeros(lda.num_topics)
    doc = bow_mail_body(text, nlp)
    bow = dic.doc2bow(doc)
    topic_score_tups = lda.get_document_topics(bow)
    for ind, score in topic_score_tups:
        topic_score[ind] = score
    return topic_score


print "lda loaded"
# save all unique sender names
all_senders = list(set(dataset['sender'].tolist()))

# create address book with frequency information for each user by topics !
address_books = {}
i = 0
# we first put a dictionnary of occurence for each recipients in address_books
# so we got dictionaries in a dictionary
for tupl in dataset.itertuples():

    sender = tupl.sender

    topic_score = get_lda_score(tupl.body, lda, nlp, id2word)
    recipients = tupl.recipients
    # if sender already have an address boo update it
    if sender in address_books:
        topical_rec_occ = address_books[sender]
    else:  # else strat from scratch
        topical_rec_occ = {}
    for rec in recipients:
        if rec in topical_rec_occ:
            topical_rec_occ[rec] += topic_score
        else:
            topical_rec_occ[rec] = topic_score

    # save
    address_books[sender] = topical_rec_occ
# now replace the dictionaries in the dic address_books by lists for each topic
for sender in address_books.keys():
    topical_rec_occ = address_books[sender]
    list_top = []
    topical_rec_occ = topical_rec_occ.items()

    for i in range(n_topics):
        # extract only the frequancies for the topic we want
        list_top.append(map(lambda m: (m[0], m[1][i]), topical_rec_occ))

        # order by frequency
        list_top[i] = sorted(list_top[i], key=operator.itemgetter(1),
                             reverse=True)
    address_books[sender] = list_top

print "Frequency computed"

#############
# baseline #
#############


def predict(pd_dataset, k=10):
    '''
    args :
        - pd_dataset, pandas dataset as formatted by load_dataset
        - k int number of recipients
    return :
        list of tuple int, list of string : [(mid, [email1, email2]), ...]
    '''
    topic_freq_pred = []
    for tupl in pd_dataset.itertuples():
        mid = tupl.mid
        sender = tupl.sender

        topic_score = get_lda_score(tupl.body, lda, nlp, id2word)

        # merge list with frequencies into a single dic
        merge_dic = {}
        for topic in range(n_topics):
            for name, freq in address_books[sender][topic]:
                if name in merge_dic:
                    merge_dic[name] += freq * topic_score[topic]
                else:
                    merge_dic[name] = freq * topic_score[topic]
        # we sort and take the k bests
        preds = sorted(merge_dic.items(), key=operator.itemgetter(1),
                       reverse=True)[:k]
        preds = [item[0] for item in preds]
        topic_freq_pred.append((mid, preds))
    return topic_freq_pred


topic_freq_pred = predict(dataset)

predicted = [preds for mid, preds in topic_freq_pred]

mids = [mid for mid, preds in topic_freq_pred]
actual = dataset[dataset['mid'] == mids]['recipients'].tolist()

print mapk(actual, predicted)

#################################################
# write predictions in proper format for Kaggle #
#################################################

path_to_results = ""

with open(path_to_results + 'predictions_frequency_topic.txt', 'wb') as f:
    f.write('mid,recipients' + '\n')
    for mid, preds in topic_freq_pred:
        f.write(str(mid) + ',' + ' '.join(preds) + '\n')
