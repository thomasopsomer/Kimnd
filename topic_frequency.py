import random
import operator
import os

import pandas as pd
import numpy as np  # FIXME
from collections import Counter
from gensim import corpora, models

from utils import load_dataset


path_to_data = "data"
lda_path = "LDA/LDA_data_lda"
lda_corpus = "LDA/LDA_data_corp"

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
    topic_score = np.abs(np.empty(10))  # FIXME !
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

# number of recipients to predict
k = 10

topic_freq_pred = []
for tupl in testset.itertuples():
    mid = tupl.mid
    sender = tupl.sender
    topic_score = lda(tupl.body)

    # merge list with frequencies into a single dic
    merge_dic = {}
    for topic in range(n_topics):
        for name, freq in address_books[sender][topic]:
            if name in merged_list:
                merge_dic[name] += freq * topic_score[topic]
            else:
                merge_dic[name] = freq * topic_score[topic]
    # we sort and take the k bests
    preds = sorted(merged_dic.items(), key=operator.itemgetter(1),
                   reverse=True)[:k]
    topic_freq_pred.append((mid, preds))

#################################################
# write predictions in proper format for Kaggle #
#################################################

path_to_results = ""

with open(path_to_results + 'predictions_frequency_topic.txt', 'wb') as f:
    f.write('mid,recipients' + '\n')
    for mid, preds in topic_freq_pred:
        f.write(str(ids[index]) + ',' + ' '.join(preds) + '\n')
