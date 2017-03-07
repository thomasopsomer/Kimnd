import random
import operator
import pandas as pd
import numpy as np
from collections import Counter
import utils


# Get the address book of each user
def address_book_users(df):
    book = df.groupby("sender").recipients.sum()
    return


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
    df_flat = df_flat[df_flat["time"]<time]
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


### 3. More Recent Outgoing Percentage ###
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
    more_recents = more_recents.rename(columns = {"sender":"user", "recipient": "contact", "last_time": "OUT_last_time"})
    # Take only the columns that we want
    more_recents = more_recents[["user", "contact", "OUT_last_time", "OUT_ratio_recent"]]
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
    more_recents = more_recents.rename(columns = {"recipient":"user", "sender": "contact", "last_time": "IN_last_time"})
    more_recents = more_recents[["user", "contact", "IN_last_time", "IN_ratio_recent"]]
    return more_recents


### 5. Combining all features ###
def get_features_out_in(df_flat, time):
    print "Outgoing Message Percentage"
    frequencies_out = get_frequencies_outgoing(df_flat, time)
    print "Incoming Message Percentage"
    frequencies_in = get_frequencies_incoming(df_flat, time)
    print "More Recent Outgoing Percentage"
    recent_out = get_more_recent_perc_out(df_flat, time)
    print "More Recent Incoming Percentage"
    recent_in = get_more_recent_perc_in(df_flat, time)
    # Join all the DataFrames
    outgoing = frequencies_out.merge(recent_out, how="inner", on=["user", "contact"])
    incoming = frequencies_in.merge(recent_in, how="inner", on=["user", "contact"])
    time_features = outgoing.merge(incoming, how="outer", on=["user", "contact"])
    print "Processing the features"
    # List of all senders and all recipients
    senders = df_flat.sender.unique()
    recipients = df_flat.recipient.unique()
    # If the  user never sent a mail, set OUT_ratio_recent to 1 and OUT_ratio to -1
    time_features.loc[~time_features["user"].isin(senders), "OUT_ratio"] = -1
    time_features.loc[~time_features["user"].isin(senders), "OUT_ratio_recent"] = 1
    # If the user never received a mail, set IN_ratio_recent to 1 and IN_ratio to -1
    time_features.loc[~time_features["user"].isin(recipients), "IN_ratio"] = -1
    time_features.loc[~time_features["user"].isin(recipients), "IN_ratio_recent"] = 1
    # If we have a NULL value, this means that the user never sent/received a mail to/from this contact
    time_features["OUT_ratio"] = time_features["OUT_ratio"].fillna(0)
    time_features["OUT_ratio_recent"] = time_features["OUT_ratio_recent"].fillna(0)
    time_features["IN_ratio"] = time_features["IN_ratio"].fillna(0)
    time_features["IN_ratio_recent"] = time_features["IN_ratio_recent"].fillna(0)
    time_features["time"] = time
    return time_features


if __name__=="__main__":

    print "Loading the files"
    dataset_path = "data/training_set.csv"
    mail_path = "data/training_info.csv"

    train_df = utils.load_dataset(dataset_path, mail_path, train=True, flat=True)
    train_df = utils.preprocess_bodies(train_df, type="train")

    #####################
    # Temporal features #
    #####################

    print "Handling time"
    origine_time = train_df["date"].min()
    train_df["time"] = (train_df["date"] - origine_time).apply(lambda x: x.seconds)

    print "Time features extraction"
    time = train_df["time"].max() + 1;
    time_features = get_features_out_in(train_df_flat, time)
    time_features.to_csv("time_features.csv", sep=",", index=False)


    #####################
    # Textual features #
    #####################
