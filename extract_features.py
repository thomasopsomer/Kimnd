# coding: utf-8

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from ast import literal_eval
import utils
import flat_dataset
from sklearn.ensemble import RandomForestClassifier
from gow import tw_idf
from os import path
import cPickle as pkl
import temporal_features
import textual_features


# Get the address book of each user
def address_book_users(df):
    book = df.groupby("sender").recipients.sum()
    book = book.map(lambda x: set(x))
    return book


### Predicting for each user ###
def add_recipients(df, all_emails):
    """all_emails: all ID contacts of all users"""
    user = df["sender"].iloc[0] # ID of the user
    emails = all_emails[user]
    df["emails"] = str(list(emails))
    df["emails"] = df["emails"].map(literal_eval)
    return df


def get_test_set(test_user, features, all_emails, reg):
    """test_user: the DataFrame of the test for a specific sender
    features: the features extracted
    emails: list of all email
    reg: the trained predictor"""
    # Create a dataset with all the possible combinations (userID, mid, contactID)
    test_user = add_recipients(test_user, contacts)
    test_user = utils.flatmap(test_user, "emails", "recipient", np.string0)

    # Some renaming
    test_user = test_user[["sender", "recipient", "mid"]]
    return test_user


if __name__ == "__main__":

    print "Loading the files"
    dataset_path = "data/training_set.csv"
    dataset_path2 = "data/test_set.csv"
    mail_path = "data/training_info.csv"
    mail_path2 = "data/test_info.csv"

    train_df = utils.load_dataset(dataset_path, mail_path, train=True, flat=True)
    train_df_not_flat = utils.load_dataset(dataset_path, mail_path, train=True, flat=False)
    test_df = utils.load_dataset(dataset_path2, mail_path2, train=False)

    ## TEST
    # train_df_not_flat = train_df_not_flat[:5000]
    # train_df = train_df[:42508]

    print "Preprocessing messages"
    train_df_not_flat = utils.preprocess_bodies(train_df_not_flat, type="train")
    test_df = utils.preprocess_bodies(test_df, type="test")

    print "Extracting global text features"
    idf_path = "idf.pkl"
    if path.exists(idf_path):
        idf = pkl.load(open(idf_path, "rb"))
        id2word = pkl.load(open("id2word.pkl", "rb"))
        texts = list(train_df_not_flat["tokens"])
        avg_len = sum(len(terms) for terms in texts) / len(texts)
    else:
        idf, id2word, avg_len = textual_features.get_global_text_features(list(train_df_not_flat["tokens"]))
        with open(idf_path, "w") as f:
            pkl.dump(idf, f)
        with open("id2word.pkl", "w") as f:
            pkl.dump(id2word, f)

    #####################
    # Temporal features #
    #####################

    time_path = "time_features.csv"
    if path.exists(time_path):
        print "Getting time features"
        time_features = pd.read_csv("time_features.csv")
    else:
        print "Handling time"
        origine_time = train_df["date"].min()
        train_df["time"] = (train_df["date"] - origine_time).apply(lambda x: x.seconds)

        print "Time features extraction"
        time = train_df["time"].max() + 1;
        time_features = temporal_features.get_features_out_in(train_df, time)
        time_features.to_csv("time_features.csv", sep=",", index=False)

    #####################
    # Textual features #
    #####################

    print "Computing and storing tw-idf of all messages"
    pickle_path = "twidf_dico_train.pkl"
    if path.exists(pickle_path):
        twidf_dico = pkl.load(open(pickle_path, "rb"))
    else:
        twidf_dico = {}
        for ind, row in train_df_not_flat.iterrows():
            if (ind+1) % 1000 == 0: print "Processesed ", ind+1
            mid = row["mid"]
            tokens = row["tokens"]
            twidf_dico[mid] = tw_idf(tokens, idf, id2word, avg_len)
        with open(pickle_path, "w") as f:
            pkl.dump(twidf_dico, f)

    pickle_path = "twidf_dico_test.pkl"
    if path.exists(pickle_path):
        twidf_dico_test = pkl.load(open(pickle_path, "rb"))
    else:
        twidf_dico_test = {}
        for ind, row in test_df.iterrows():
            if (ind+1) % 1000 == 0: print "Processesed ", ind+1
            mid = row["mid"]
            tokens = row["tokens"]
            twidf_dico_test[mid] = tw_idf(tokens, idf, id2word, avg_len)
        with open(pickle_path, "w") as f:
            pkl.dump(twidf_dico_test, f)

    print "Getting the averages dictionaries for outgoind and incoming messages"
    # Computes the average tw idf vector (incoming)
    dict_tuple_mids_in = train_df.groupby(["recipient", "sender"])["mid"].apply(list).to_dict()
    for tupl in dict_tuple_mids_in.keys():
        dict_tuple_mids_in[tupl] = np.average(np.array([twidf_dico[m].toarray() for m in dict_tuple_mids_in[tupl]]), axis=0)
        dict_tuple_mids_in[tupl] = csr_matrix(dict_tuple_mids_in[tupl])
    # train_df['incoming_txt'] = textual_features.incoming_text_similarity_new(
    #     train_df, twidf_dico, dict_tuple_mids_in)

    # Computes the average tw idf vector (outgoing)
    dict_tuple_mids_out = train_df.groupby(["sender", "recipient"])["mid"].apply(list).to_dict()
    for tupl in dict_tuple_mids_out.keys():
        dict_tuple_mids_out[tupl] = np.average(np.array([twidf_dico[m].toarray() for m in dict_tuple_mids_out[tupl]]), axis=0)
        dict_tuple_mids_out[tupl] = csr_matrix(dict_tuple_mids_out[tupl])
    # train_df['outgoing_txt'] = textual_features.outgoing_text_similarity_new(
    #     train_df, twidf_dico, dict_tuple_mids_out)

    # text_features = train_df[["sender", "recipient", "mid", "outgoing_txt", "incoming_txt"]]

    ###############
    # Classifier #
    ###############

    print "Preparing for the ranking"
    # Extract all the emails of the database
    emails = set(train_df["sender"]).union(set(train_df["recipient"]))

    # Get all contacts for each user
    contacts = time_features.groupby("user").contact.apply(set)

    print "Generating positive and negative pairs"
    # Get the positive and negative pairs for the classifier
    pairs_train = flat_dataset.make_flat_dataset(train_df_not_flat, contacts, 1.0, num_cores=4)

    # Adding textual features
    print "Textual features for the pairs"
    pairs_train['outgoing_txt'] = textual_features.outgoing_text_similarity_new(
        pairs_train, twidf_dico, dict_tuple_mids_out)
    pairs_train['incoming_txt'] = textual_features.incoming_text_similarity_new(pairs_train, twidf_dico, dict_tuple_mids_in)

    # Renaming
    pairs_train = pairs_train.rename(columns={"sender":"user", "recipient": "contact"})
    pairs_train = pairs_train[["user", "contact", "mid", "incoming_txt", "outgoing_txt", "label"]]

    print "Training"
    # Train arrays
    X_train = pairs_train.merge(time_features, how="left", on=["contact", "user"])
    X_train = X_train.fillna(0)
    y_train = X_train["label"].values
    X_train = X_train.set_index(["contact", "mid", "user"])
    X_train = X_train.drop(["label"], axis=1)
    X_train = X_train.values

    # Training
    clf = RandomForestClassifier(n_estimators=50, random_state=42, oob_score=True, n_jobs=-1)
    clf.fit(X_train, y_train)
    print clf.oob_score_

    print "Getting the test set ready"
    # Prediction
    test_pairs = test_df.groupby("sender").apply(lambda test_user: get_test_set(test_user, time_features, contacts, clf))
    test_pairs = test_pairs.reset_index(drop=True)

    print "Adding textual features to the test set"
    test_pairs['outgoing_txt'] = textual_features.outgoing_text_similarity_new(
        test_pairs, twidf_dico_test, dict_tuple_mids_out)
    test_pairs['incoming_txt'] = textual_features.incoming_text_similarity_new(
        test_pairs, twidf_dico_test, dict_tuple_mids_in)

    test_pairs = test_pairs.rename(columns={"sender":"user", "recipient": "contact"})

    # Getting the arrays for the prediction
    X_test = test_pairs.merge(time_features, how="left", on=["contact", "user"])
    X_test = X_test.fillna(0)
    X_test = X_test.set_index(["contact", "mid", "user"])
    test_index = X_test.index
    X_test = X_test.values

    print "Predictions"
    # Predictions
    pred = clf.predict_proba(X_test)[:,1]
    pred = pd.DataFrame(pred, columns=["pred"], index=test_index).reset_index()
    
    # We take the top 10 for each mail
    res = pred.groupby("mid").apply(lambda row: row.sort_values(
        by="pred", ascending=False).head(10)).reset_index(drop=True)
    res = res[["mid", "contact"]]
    res = res.groupby("mid").contact.apply(list).reset_index()
    res["contact"] = res.contact.map(lambda x: ' '.join(x))
    res.to_csv("results_time_text_clf.csv", header=["mid", "recipients"], index=False)
