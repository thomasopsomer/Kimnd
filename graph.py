import operator
import pandas as pd
from collections import Counter
import networkx as nx
import os
import pdb

path_to_data = os.path.join(os.getcwd(), 'data/')

##########################
# load some of the files #
##########################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

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
all_senders = emails_ids_per_sender.keys()

# create address book with frequency information for each user
address_books = {}
i = 0

# create graph
graph = nx.DiGraph()

for sender, ids in emails_ids_per_sender.iteritems():
    recs_temp = []
    dates_dict = {}
    for my_id in ids:
        row_info = training_info[training_info['mid'] == int(my_id)]
        # store recipients' mails
        recipients = row_info['recipients'].tolist()
        recipients = recipients[0].split(' ')
        # keep only legitimate email addresses
        recipients = [rec for rec in recipients if '@' in rec]
        recs_temp.append(recipients)
        # store dates
        date = row_info['date'].values[0]  # change format later
        for rec in recipients:
            if rec not in dates_dict:
                dates_dict[rec] = [date]
            else:
                dates_dict[rec].append(date)
    # flatten
    recs_temp = [elt for sublist in recs_temp for elt in sublist]
    # compute recipient counts
    rec_occ = dict(Counter(recs_temp))
    # order by frequency
    sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse=True)
    # save
    address_books[sender] = sorted_rec_occ
    # add edges to graph: from sender to recipient, attributes: dates and frequency
    for rec, occ in rec_occ.iteritems():
        graph.add_edge(sender, rec, occ=occ)
    for rec, date in dates_dict.iteritems():
        graph[sender][rec].update(date=date)

    if i % 10 == 0:
        print i
    i += 1

# print edges
graph.edges(data=True)
