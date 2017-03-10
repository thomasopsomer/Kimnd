import random
import operator
import pandas as pd
import numpy as np
from collections import Counter
from ast import literal_eval
import utils
import flat_dataset
from sklearn.ensemble import RandomForestRegressor
import textual_features
import time_features_


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


def predict(test_user, features, all_emails, reg):
    """test_user: the DataFrame of the test for a specific sender
    features: the features extracted
    emails: list of all email
    reg: the trained predictor"""
    # Create a dataset with all the possible combinations (userID, mid, contactID)
    test_user = add_recipients(test_user, contacts)
    test_user = utils.flatmap(test_user, "emails", "contact", np.string0)

    # Some renaming
    test_user = test_user.rename(columns={"sender":"user"})
    test_user = test_user[["user", "contact", "mid"]]
    return test_user


if __name__=="__main__":

    print "Loading the files"
    dataset_path = "data/training_set.csv"
    dataset_path2 = "data/test_set.csv"
    mail_path = "data/training_info.csv"
    mail_path2 = "data/test_info.csv"

    train_df = utils.load_dataset(dataset_path, mail_path, train=True, flat=True)
    train_df_not_flat = utils.load_dataset(dataset_path, mail_path, train=True, flat=False)
    test_df = utils.load_dataset(dataset_path2, mail_path2, train=False)

    print "Preprocessing messages"
    train_df_not_flat = utils.preprocess_bodies(train_df_not_flat, type="train")
    test_df = utils.preprocess_bodies(test_df, type="test")

    print "Extracting global text features"
    idf, id2word, avg_len = textual_features.get_global_text_features(list(train_df_not_flat["tokens"]))

    #####################
    # Temporal features #
    #####################

    try:
        print "Getting time features"
        time_features = pd.read_csv("time_features.csv")
    except Exception as e:
        print "Handling time"
        origine_time = train_df["date"].min()
        train_df["time"] = (train_df["date"] - origine_time).apply(lambda x: x.seconds)

        print "Time features extraction"
        time = train_df["time"].max() + 1;
        time_features = time_features_.get_features_out_in(train_df, time)
        time_features.to_csv("time_features.csv", sep=",", index=False)


    #####################
    # Textual features #
    #####################

    def get_outgoing_mid(mid, df, list_sender, idf, id2word, avg_len, n):
        df_all_outgoing = pd.DataFrame(columns=['mid', 'user', 'contact', 'outgoing_text'])
        for user in list_sender:
            df_all_outgoing.append(textual_features.outgoing_text_similarity(
                df, mid, user, idf, id2word, avg_len, n)
            )
        return df_all_outgoing

    def get_outgoing_all(df, list_sender, idf, id2word, avg_len, n):
        df_all_outgoing = df.mid.map(lambda mid: get_outgoing_mid(mid, df, list_sender, idf, id2word, avg_len, n))
        return df_all_outgoing

    n = 5  # number of similar messages
    list_sender = np.unique(train_df_not_flat['sender'].tolist())
    list_recipients = np.unique(train_df['recipient'].tolist())
    df_all_outgoing = flat_dataset.parallelize_dataframe(
        train_df_not_flat, get_outgoing_all, 4, log=False, list_sender=list_sender, idf=idf, id2word=id2word, avg_len=avg_len, n)

    # for ind, row in train_df_not_flat.iterrows():
    #     print(ind)
    #     mid = row['mid']
    #     df_all_outgoing = pd.DataFrame(columns=['mid', 'user', 'contact', 'outgoing_text'])
    #     for user in list_sender:
    #         df_all_outgoing.append(textual_features.outgoing_text_similarity(
    #             train_df_not_flat, mid, user, idf, id2word, avg_len, n)
    #         )
    #     df_all_incoming = pd.DataFrame(columns=['mid', 'user', 'contact', 'incoming_text'])

    #     for user in list_recipients:
    #         print(user)
    #         df_all_incoming.append(textual_features.incoming_text_similarity(
    #             train_df_not_flat, mid, user, idf, id2word, avg_len, n)
    #         )


    ###############
    # Classifier #
    ###############

    print "Preparing for the ranking"
    # Extract all the emails of the database and attribute an unique id to it
    emails = set(train_df["sender"]).union(set(train_df["recipient"]))

    import pdb; pdb.set_trace()

    # Get all contacts for each user
    contacts = time_features.groupby("user").contact.apply(set)

    print "Generating positive and negative pairs"
    # Get the positive and negative pairs for the classifier
    pairs_train = flat_dataset.make_flat_dataset(train_df_not_flat, contacts, 1.0, num_cores=4)
    pairs_train = pairs_train.rename(columns={"sender":"user", "recipient": "contact"})
    pairs_train = pairs_train[["user", "contact", "mid", "label"]]

    print "Training"
    # Train arrays
    X_train = pairs_train.merge(time_features, how="left", on=["contact", "user"])
    X_train = X_train.fillna(0)
    y_train = X_train["label"].values
    X_train = X_train.set_index(["contact", "mid", "user"])
    X_train = X_train.drop(["label"], axis=1)
    X_train = X_train.values

    # Training
    reg = RandomForestRegressor(n_estimators=500, random_state=42, oob_score=True)
    reg.fit(X_train, y_train)

    print "Getting the test set ready"
    # Prediction
    test_pairs = test_df.groupby("sender").apply(
        lambda test_user: predict(test_user, time_features, contacts, reg))
    test_pairs = test_pairs.reset_index(drop=True)

    # Getting the arrays for the prediction
    X_test = test_pairs.merge(time_features, how="left", on=["contact", "user"])
    X_test = X_test.fillna(0)
    X_test = X_test.set_index(["contact", "mid", "user"])
    test_index = X_test.index
    X_test = X_test.values

    print "Predictions"
    # Predictions
    pred = reg.predict(X_test)
    pred = pd.DataFrame(pred, columns=["pred"], index=test_index).reset_index()
    
    # We take the top 10 for each mail
    res = pred.groupby("mid").apply(lambda row: row.sort_values(by="pred", ascending=False).head(10)).reset_index(drop=True)
    res = res[["mid", "contact"]]
    res = res.groupby("mid").contact.apply(list).reset_index()
    res["contact"] = res.contact.map(lambda x: ' '.join(x))
    res.to_csv("results_time.csv", header=["mid", "recipients"], index=False)
