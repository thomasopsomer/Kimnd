import random
import operator
import os

from copy import deepcopy
import pandas as pd
from collections import Counter

path_to_data = "data/"

##########################
# load some of the files #
##########################

training = pd.read_csv(os.path.join(path_to_data, 'training_set.csv'),
                       sep=',', header=0)

training_info = pd.read_csv(os.path.join(path_to_data, 'training_info.csv'),
                            sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)

################################
# create some handy structures #
################################

# convert training set to dictionary
emails_ids_per_sender = {}
for index, series in training.iterrows():
    row = series.tolist()
    sender = row[0]
    ids = row[1:][0].split(' ')
    emails_ids_per_sender[sender] = ids

# save all unique sender names
all_senders = sorted(emails_ids_per_sender.keys())

# create address book with frequency information for each user
address_books = {}
i = 0

for sender, ids in emails_ids_per_sender.iteritems():
    recs_temp = []
    for my_id in ids:
        recipients = training_info[
            training_info['mid'] == int(my_id)]['recipients'].tolist()
        recipients = recipients[0].split(' ')
        # keep only legitimate email addresses
        recipients = [rec for rec in recipients if '@' in rec]
        recs_temp.append(recipients)
    # flatten
    recs_temp = [elt for sublist in recs_temp for elt in sublist]
    # compute recipient counts
    rec_occ = dict(Counter(recs_temp))
    # order by frequency
    sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1),
                            reverse=True)
    # save
    address_books[sender] = sorted_rec_occ

    if i % 10 == 0:
        print i
    i += 1

# save all unique recipient names
all_recs = sorted(list(set([elt[0] for sublist in
                            address_books.values() for elt in sublist])))

# save all unique user names
all_users = []
all_users.extend(all_senders)
all_users.extend(all_recs)
all_users = list(set(all_users))

#############
# baselines #
#############

train_info2 = deepcopy(training_info)
for sender, ids in emails_ids_per_sender.iteritems():

    for my_id in ids:
        recs_temp = ""
        recipients = training_info[
            training_info['mid'] == int(my_id)]['recipients'].tolist()
        recipients = recipients[0].split(' ')
        # keep only legitimate email addresses
        recipients = [rec for rec in recipients if '@' in rec]
        rec_id = "".join([str(all_recs.index(elt))+',' for elt in recipients])
        train_info2.loc[
            training_info['mid'] == int(my_id), 'recipients'] = rec_id
        train_info2.loc[
            training_info['mid'] == int(my_id),
            'sender'] = all_senders.index(sender)

train_info2.to_csv('info2.csv')
