#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import multiprocessing as mp
from math import ceil
import numpy as np
from functools import partial


def sample_neg_recipient(recipients, emails, percent=0.3):
    """
    recipient: list of recipient emails
    emails: set of all emails
    percent: 0-1 float for percentage of negative samples
    """
    n_rec = len(recipients)
    n_neg = int(ceil(percent * n_rec))
    # sample randomly n_neg fake recipient among the negative emails
    neg_emails = list(emails.difference(recipients))
    neg_samples = np.random.choice(neg_emails, n_neg)
    return neg_samples


def sample_neg_rec_df(df, emails, percent=0.3):
    """ """
    df["negs"] = df.recipients.map(
        lambda x: sample_neg_recipient(
            x, emails, percent=percent))
    return df


def parallelize_dataframe(df, func, num_cores, **kwargs):
    """
    Function to parallelize a function over rows of a dataset :)
    """
    df_split = np.array_split(df, num_cores)
    pool = mp.Pool(num_cores)
    partial_f = partial(func, **kwargs)
    try:
        print 'starting the pool map'
        df = pd.concat(pool.map(partial_f, df_split))
        pool.close()
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
        print 'joining pool processes'
        pool.join()
        print 'join complete'
    print 'the end'
    return df


def make_flat_dataset(df, emails, mail2id, fake_percent, num_cores=4):
	"""
	df is supposed to be the df obtained with load_dataset :)
	mail2id is supposed to be a dictionnary of emails to their ids
	"""
    # create fake paires randomly
    df_neg = parallelize_dataframe(
        df, sample_neg_rec_df, num_cores=num_cores, emails=emails,
        percent=fake_percent)

    # flatten the recipients
    df_flat_rec = flatmap(df, "recipients", "recipient")
    df_flat_neg = flatmap(
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



