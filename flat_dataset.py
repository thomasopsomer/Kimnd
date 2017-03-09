import pandas as pd
import multiprocessing as mp
from math import ceil
from ast import literal_eval
import numpy as np
from functools import partial
import utils


def sample_neg_recipient(row, all_emails, percent=0.3):
    """
    row: one row of the non flat dataframe
    emails: set of email contacts for each user
    percent: 0-1 float for percentage of negative samples
    """
    recipients = set(row["recipients"])
    emails = all_emails[row["sender"]]
    n_rec = len(recipients)
    n_neg = int(ceil(percent * n_rec))
    # sample randomly n_neg fake recipient among the negative emails
    neg_emails = list(emails.difference(recipients))
    n_neg = min(len(neg_emails), n_neg) # We can't exceed the number of users
    if len(neg_emails) == 0:
        return str([])
    neg_samples = np.random.choice(neg_emails, n_neg, replace=False)
    return str(list(neg_samples))


def sample_neg_rec_df(df, all_emails, percent=0.3):
    """all_emails: Series of all contacts of each user"""
    df["negs"] = df.apply(
        lambda row: sample_neg_recipient(row, all_emails, percent=0.3),
        axis=1).apply(literal_eval)
    return df


# Parallelization
def parallelize_dataframe(df, func, num_cores, log=False, **kwargs):
    """
    Function to parallelize a function over rows of a dataset :)
    """
    df_split = np.array_split(df, num_cores)
    pool = mp.Pool(num_cores)
    partial_f = partial(func, **kwargs)
    try:
        if log:
            print 'starting the pool map'
        df = pd.concat(pool.map(partial_f, df_split))
        pool.close()
        if log:
            print 'pool map complete'
    except KeyboardInterrupt:
        print 'got ^C while pool mapping, terminating the pool'
        pool.terminate()
        print 'pool is terminated'
    except Exception, e:
        print 'got exception: %r, terminating the pool' % (e,)
        pool.terminate()
        print 'pool is terminated'
    finally:
        if log:
            print 'joining pool processes'
        pool.join()
        if log:
            print 'join complete'
    if log:
        print 'the end'
    return df


def make_flat_dataset(df, all_emails, mail2id, fake_percent, num_cores=4, log=False):
    """
    df is supposed to be the df obtained with load_dataset :)
    mail2id is supposed to be a dictionnary of emails to their ids
    """
    # create fake paires randomly
    df_neg = parallelize_dataframe(
        df, sample_neg_rec_df, num_cores=num_cores, all_emails=all_emails,
        percent=fake_percent)
    # flatten the recipients
    df_flat_rec = utils.flatmap(df, "recipients", "recipient")
    df_flat_neg = utils.flatmap(
        df_neg.drop("recipients", axis=1), "negs", "recipient")
    # add labels: 0 for fake recipient, 1 for others
    df_flat_rec["label"] = 1
    df_flat_neg["label"] = 0
    # concat neg and real recipient paires
    df_flat = pd.concat((df_flat_rec, df_flat_neg), axis=0)
    # alias mail with ids
    df_flat.sender = df_flat.sender.map(lambda x: mail2id[x])
    df_flat.recipient = df_flat.recipient.map(lambda x: mail2id[x])
    #
    return df_flat
