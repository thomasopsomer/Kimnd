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


# Get the address book of each user
def address_book_users(df):
    book = df.groupby("sender").recipients.sum()
    book = book.map(lambda x: set(x))
    return book


### 1. Outgoing Message Percentage ###
def get_frequencies_outgoing(df_flat, time):
    # Take only the mails sent before time
    df_flat = df_flat[df_flat["time"]<time]
    if df_flat.empty:
        return pd.DataFrame([])
    # Count the number of mails for each pair of (sender, recipient)
    frequencies_recipient = pd.DataFrame(df_flat.groupby(["sender", "recipient"]).mid.nunique()).reset_index()
    # Count the total mails for each user
    frequencies_all = pd.DataFrame(df_flat.groupby("sender").mid.nunique()).reset_index()
    # Rename the columns for better readability
    frequencies_recipient.columns = ["sender", "recipient", "OUT_cnt_mess_recipient"]
    frequencies_all.columns = ["sender", "OUT_cnt_mess_all"]
    # Perform an inner join on the column "sender" between the 2 dataframes
    frequencies = frequencies_recipient.merge(frequencies_all, how="inner", on="sender")
    # OUT_ratio is the desired final feature
    frequencies["OUT_ratio"] = frequencies["OUT_cnt_mess_recipient"] / frequencies["OUT_cnt_mess_all"]
    frequencies = frequencies.rename(columns = {"sender":"user", "recipient": "contact"}) # user and contact as in the paper
    frequencies = frequencies[["user", "contact", "OUT_ratio"]]
    return frequencies


### 2. Incoming Message Percentage ###
def get_frequencies_incoming(df_flat, time):
    # Same operations as the previous function
    # This time, the user is the recipient
    df_flat = df_flat[df_flat["time"] < time]
    if df_flat.empty:
        return pd.DataFrame([])
    frequencies_sender = pd.DataFrame(df_flat.groupby(["recipient", "sender"]).mid.nunique()).reset_index()
    frequencies_all = pd.DataFrame(df_flat.groupby("recipient").mid.nunique()).reset_index()
    frequencies_sender.columns = ["recipient", "sender", "IN_cnt_mess_sender"]
    frequencies_all.columns = ["recipient", "IN_cnt_mess_all"]
    frequencies = frequencies_sender.merge(frequencies_all, how="inner", on="recipient")
    frequencies["IN_ratio"] = frequencies["IN_cnt_mess_sender"] / frequencies["IN_cnt_mess_all"]
    frequencies = frequencies.rename(columns = {"recipient":"user", "sender": "contact"})
    frequencies = frequencies[["user", "contact", "IN_ratio"]]
    return frequencies


# ## 3. More Recent Outgoing Percentage ###
def get_last_time(df_flat):
    # Get last time email was sent to a recipient from a sender
    # At the end we have one row for each couple (sender, recipient)
    # and the value for the last time a mail was sent between them
    time_last_sent = df_flat.groupby(["sender", "recipient"]).time.max().reset_index()
    time_last_sent.columns = ["sender", "recipient", "last_time"]
    return time_last_sent


def get_cnt_mail_out(df_flat, sender, begin, end):
    # Given a sender and a period of time [begin, end] count the mails sent within this period
    # Filter only the mails sent in a time t between begin and end
    df_flat = df_flat[df_flat["time"]>=begin]
    df_flat = df_flat[df_flat["time"]<end]
    if df_flat.empty:
        return 0
    # Count the mails sent in this period for each sender
    count_out = df_flat.groupby("sender").mid.nunique()
    return count_out[sender]


def get_more_recent_perc_out(df_flat, time):
    # Computes the More Recent Outgoing Percentage as in the paper
    alpha = 2
    # For each couple (user, sender) extract the last time a mail was sent between them
    more_recents = get_last_time(df_flat)
    # Apply the function get_cnt_mail_out row-wise
    # to take into account each possible combination of (user, sender)
    more_recents["OUT_cnt_mess_recent"] = more_recents.apply(lambda row:
        get_cnt_mail_out(df_flat, row["sender"], row["last_time"], time), axis=1)
    # Count the total messages sent by the sender
    cnt_all_mess = pd.DataFrame(df_flat.groupby("sender").mid.nunique()).reset_index()
    cnt_all_mess.columns = ["sender", "OUT_cnt_mess_all"]
    # Perform an inner join on the column "sender" between the 2 datasets
    more_recents = more_recents.merge(cnt_all_mess, how="inner", on="sender")
    # The desired feature
    more_recents["OUT_ratio_recent"] = more_recents["OUT_cnt_mess_recent"] / (alpha*more_recents["OUT_cnt_mess_all"])
    more_recents = more_recents.rename(columns = {"sender":"user", "recipient": "contact"})
    # Take only the columns that we want
    more_recents = more_recents[["user", "contact", "OUT_ratio_recent"]]
    return more_recents


### 4. More Recent Incoming Percentage ###
# Same operations as for the out feature
def get_cnt_mail_in(df_flat, recipient, begin, end):
    df_flat = df_flat[df_flat["time"]>=begin]
    df_flat = df_flat[df_flat["time"]<end]
    if df_flat.empty:
        return 0
    count_out = df_flat.groupby("recipient").mid.nunique()
    return count_out[recipient]


def get_more_recent_perc_in(df_flat, time):
    alpha = 2
    more_recents = get_last_time(df_flat)
    more_recents["IN_cnt_mess_recent"] = more_recents.apply(lambda row:
        get_cnt_mail_in(df_flat, row["recipient"], row["last_time"], time), axis=1)
    cnt_all_mess = pd.DataFrame(df_flat.groupby("recipient").mid.nunique()).reset_index()
    cnt_all_mess.columns = ["recipient", "IN_cnt_mess_all"]
    more_recents = more_recents.merge(cnt_all_mess, how="inner", on="recipient")
    more_recents["IN_ratio_recent"] = more_recents["IN_cnt_mess_recent"] / (alpha*more_recents["IN_cnt_mess_all"])
    more_recents = more_recents.rename(columns = {"recipient":"user", "sender": "contact"})
    more_recents = more_recents[["user", "contact", "IN_ratio_recent"]]
    return more_recents


### 5. Combining all features ###
def get_features_out_in(df_flat, time):
    print "Outgoing Message Percentage"
    frequencies_out = get_frequencies_outgoing(df_flat, time)
    print "Incoming Message Percentage"
    frequencies_in = get_frequencies_incoming(df_flat, time)
    print "More Recent Outgoing Percentage"
    recent_out = get_more_recent_perc_out(df_flat, time)
    #recent_out = flat_dataset.parallelize_dataframe(df_flat, get_more_recent_perc_out, num_cores=4, time=time)
    print "More Recent Incoming Percentage"
    recent_in = get_more_recent_perc_in(df_flat, time)
    #recent_in = flat_dataset.parallelize_dataframe(df_flat, get_more_recent_perc_in, num_cores=4, time=time)
    # Join all the DataFrames
    outgoing = frequencies_out.merge(recent_out, how="inner", on=["user", "contact"])
    incoming = frequencies_in.merge(recent_in, how="inner", on=["user", "contact"])
    time_features = outgoing.merge(incoming, how="outer", on=["user", "contact"])
    print "Processing the features"
    # List of all senders and all recipients
    senders = df_flat.sender.unique()
    recipients = df_flat.recipient.unique()
    # If the  user never sent a mail, set OUT_ratio_recent to 1 and OUT_ratio to 1
    time_features.loc[~time_features["user"].isin(senders), "OUT_ratio"] = 1
    time_features.loc[~time_features["user"].isin(senders), "OUT_ratio_recent"] = 1
    # If the user never received a mail, set IN_ratio_recent to 1 and IN_ratio to 1
    time_features.loc[~time_features["user"].isin(recipients), "IN_ratio"] = 1
    time_features.loc[~time_features["user"].isin(recipients), "IN_ratio_recent"] = 1
    # If we have a NULL value, this means that the user never sent/received a mail to/from this contact
    time_features["OUT_ratio"] = time_features["OUT_ratio"].fillna(0)
    time_features["OUT_ratio_recent"] = time_features["OUT_ratio_recent"].fillna(0)
    time_features["IN_ratio"] = time_features["IN_ratio"].fillna(0)
    time_features["IN_ratio_recent"] = time_features["IN_ratio_recent"].fillna(0)
    return time_features


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
    train_df = utils.preprocess_bodies(train_df, type="train")
    test_df = utils.preprocess_bodies(test_df, type="test")

    print "Extracting global text features"
    idf, id2word, avg_len = textual_features.get_global_text_features(list(train_df["tokens"]))

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
        time_features = get_features_out_in(train_df, time)
        time_features.to_csv("time_features.csv", sep=",", index=False)


    #####################
    # Textual features #
    #####################

    n = 5  # number of similar messages
    list_sender = np.unique(train_df_not_flat['sender'].tolist())
    list_recipients = np.unique(train_df['recipient'].tolist())
    for ind, row in train_df_not_flat.iterrows():
        mid = row['mid']
        df_all_outgoing = pd.DataFrame(columns=['mid', 'user', 'contact', 'outgoing text'])
        for user in list_sender:
            df_all_outgoing.append(textual_features.outgoing_text_similarity(
                train_df_not_flat, mid, user, idf, id2word, avg_len, 5)
            )





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
