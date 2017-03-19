#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import string
from os import path

import pandas as pd
import cPickle as pkl
import numpy as np
from spacy_utils import get_custom_spacy

from data.stopwords import extendedstopwords
from nltk.tokenize import sent_tokenize
from gensim.utils import any2unicode, deaccent
import multiprocessing as mp
from functools import partial


# utils for loading and preprocessing dataset

try:
    import regex
    REGEX = True
    lower_upper_pat = regex.compile("(?<=[a-z])(?=[A-Z])",
                                    flags=regex.VERSION1)
    number_letter_pat = regex.compile("(?<=[0-9])(?=[a-zA-Z])",
                                      flags=regex.VERSION1)
except ImportError:
    REGEX = False
not_in_list = [
    "_", "--", "*", "/", ":", "=", "(", "]", ")", "#", "|", "@", "+"]


def flatmap(df, col, new_col_name, new_col_type=None):
    """perform flatmap or 'explode' operation on column 'col'"""
    res = df[col].apply(pd.Series).unstack().reset_index().dropna()
    res = res[["level_1", 0]].set_index("level_1") \
        .rename(columns={0: new_col_name})
    res = res.merge(df[[c for c in df.columns if c != col]],
                    left_index=True, right_index=True, how="left")
    if new_col_type is not None:
        res[new_col_name] = res[new_col_name].map(new_col_type)
    return res


def load_dataset(dataset_path="data/training_set.csv",
                 mail_path="data/training_info.csv",
                 train=True, flat=False):
    """
    Load  the dataset and perform some cleaning (wrong dates), mutliple
    occurences and Merge with content of the second dataset to have everything
    under a dataframe
    if the flat option is set to true
            Explode the mail ids (mids)

    Deal with the date issues and set the date to datetime
    """

    set_df = pd.read_csv(dataset_path)
    mail_df = pd.read_csv(mail_path)
    # flatmap mail ids 'mids'
    set_df.mids = set_df.mids.str.split()
    set_df = flatmap(set_df, "mids", "mid", int)
    # merge with mail content
    set_df = set_df.merge(mail_df, on="mid")
    # fix date issue
    set_df.date = set_df.date.map(
        lambda x: x.replace("0001", "2001").replace("0002", "2002"))
    if train:
        # remove duplicates
        set_df = set_df.drop_duplicates(
            subset=["sender", "body", "date", "recipients"])
        # split recipients into list
        set_df.recipients = set_df.recipients.map(split_emails)
        # flatten recipient if needed
        if flat:
            set_df = flatmap(set_df, "recipients", "recipient", np.string_)

    set_df.date = pd.to_datetime(set_df.date)
    set_df.index = range(len(set_df.index))
    return set_df


def split_emails(string):
    """
    ARGS :
        - (string) a list of email as a string that is  ' ' delimited
    returns:
        - (list of string) list of emails containing an @
    """
    res = []
    tmp = string.split()
    keep = ""
    for part in tmp:
        if "@" in part:
            if not keep:
                res.append(part)
            else:
                res.append(keep + part)
                keep = ""
        else:
            keep += part
    return res


def split_train_dev_set(df, percent=0.2):
    """
    split dataset in train and dev set
    for each sender, we put the a percentage of the last message
    he sent in the dev set :)
    """
    train = []
    dev = []
    for k, g in df.groupby("sender"):
        n_msg = g.shape[0]
        n_dev = int(n_msg * percent)
        g = g.sort_values("date")
        g_train = g[:-n_dev]
        g_dev = g[-n_dev:]
        train.append(g_train)
        dev.append(g_dev)
    # concat all dataframe
    df_train = pd.concat(train, axis=0).sort_index()
    df_dev = pd.concat(dev, axis=0).sort_index()
    return df_train, df_dev


# Preprocessing of email content to extract Features and Cleaned text
def is_forward(txt):
    if "-----Original Message-----" in txt:
        return True
    else:
        return False

# some dude's regexes
re0 = re.compile('>')
re1 = re.compile('(Message-ID(.*?\n)*X-FileName.*?\n)|'
                 '(To:(.*?\n)*?Subject.*?\n)|'
                 '(< (Message-ID(.*?\n)*.*?X-FileName.*?\n))')
re2 = re.compile('(.+)@(.+)')  # Remove emails
# remove url
# reUrl = re.compile(
#     r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|'
#     '(?:%[0-9a-fA-F][0-9a-fA-F]))+')
re3 = re.compile('\s(-----)(.*?)(-----)\s', re.DOTALL)
re4 = re.compile('''\s(\*\*\*\*\*)(.*?)(\*\*\*\*\*)\s''', re.DOTALL)
re5 = re.compile('\s(_____)(.*?)(_____)\s', re.DOTALL)
re6 = re.compile('\n( )*-.*')
re7 = re.compile('\n( )*\d.*')
re8 = re.compile(
    '(\n( )*[\w]+($|( )*\n))|(\n( )*(\w)+(\s)+(\w)+(( )*\n)|$)|(\n( )*(\w)+'
    '(\s)+(\w)+(\s)+(\w)+(( )*\n)|$)')
re9 = re.compile('.*orwarded.*')
re10 = re.compile(
    'From.*|Sent.*|cc.*|Subject.*|Embedded.*|http.*|\w+\.\w+|'
    '.*\d\d/\d\d/\d\d\d\d.*')
re11 = re.compile(' [\d:;,.]+ ')

re_fw_pattern = r"----[-\s]*(Original|Forwarded).*Subject:"
re_fw_regex = re.compile(re_fw_pattern)


def replace_punct(s):
    # removes punctuation in words
    for c in string.punctuation:
        s = s.replace(c, "")
    return s


def drop_digits(s):
    # remove digits
    for c in range(10):
        s = s.replace(str(c), "")
    return s


def bow_mail_body(txt, nlp):
    """
    return a mail content as a Bag of words using spacy and a few regexes
    args:
        - txt: raw text
        - nlp: a spacy engine
    """
    # to unicode & get rid of accent
    txt = deaccent(any2unicode(txt))
    # split according to reply forward (get rid of "entête")
    txt = "\n".join(re_fw_regex.split(txt))
    txt = txt.replace(">", " ")
    # split sentences
    sentences = sent_tokenize(txt)
    # tokenize + lemmatize + filter ?
    bow = []
    for sent in sentences:
        if REGEX:
            sent = " ".join(lower_upper_pat.split(sent))
            sent = " ".join(number_letter_pat.split(sent))
        doc = nlp(sent, parse=False, entity=False)
        for tok in doc:
            lemma = drop_digits(replace_punct(tok.lemma_))
            if (lemma and
                not tok.is_punct and not tok.is_stop and
                lemma not in extendedstopwords and
                not tok.like_num and not tok.is_space and
                not tok.like_url and len(lemma) > 1 and
                    not any((x in tok.orth_ for x in not_in_list))):
                if tok.orth_.startswith("-") or tok.orth_.endswith("-"):
                    bow.append(lemma.replace("-", ""))
                else:
                    bow.append(lemma)
    return bow


def preprocess_bodies(dataset, type="train"):
    """
    Apply our preprocessing on a dataframe representing the whole dataset
    contains a caching system to avoid calculatuing it over and over
    """
    pickle_path = "preprocessed_data_{:s}.pkl".format(type)
    if path.exists(pickle_path):
        texts = pkl.load(open(pickle_path, "rb"))
    else:
        texts = dataset["body"]
        texts = texts.str.decode('utf-8')
        print "Loading Spacy"
        # nlp = spacy.load('en', parser=False)
        nlp = get_custom_spacy()
        print "Preprocessing mails"

        # texts = texts.apply(preprocess_mail_body, args=(nlp,))
        texts = texts.apply(bow_mail_body, args=(nlp,))
        texts = list(texts)
        with open(pickle_path, "w") as f:
            pkl.dump(texts, f)
    dataset["tokens"] = pd.Series(texts)
    dataset["tokens"] = dataset["tokens"].apply(
        lambda x: np.unique(x).tolist())  # remove duplicates
    return dataset


def extract_names(txt, nlp, n_sentences=2):
    """
    use the spacy entity engine to extract person names from a text
    args:
        - txt: raw text
        - nlp: a spacy engine
    return:
        - list of names as strings
    """
    # to unicode & get rid of accent
    txt = deaccent(any2unicode(txt))
    # split according to reply forward (get rid of "entête")
    txt = "\n".join(re_fw_regex.split(txt))
    txt = txt.replace(">", " ")
    # split sentences
    sentences = sent_tokenize(txt)
    # tokenize + lemmatize + filter ?
    bow = []
    for sent in sentences[:n_sentences]:
        if REGEX:
            sent = " ".join(lower_upper_pat.split(sent))
            sent = " ".join(number_letter_pat.split(sent))
        doc = nlp(sent, parse=False)
        for tok in doc:
            lemma = drop_digits(replace_punct(tok.lemma_))
            if (lemma and (tok.ent_type_ != 'PERSON') and
                not tok.is_punct and not tok.is_stop and
                lemma not in extendedstopwords and
                not tok.like_num and not tok.is_space and
                not tok.like_url and len(lemma) > 1 and
                    not any((x in tok.orth_ for x in not_in_list))):
                bow.append(lemma)
    return bow


def parallelize_fct(func, arg_list, num_cores, **kwargs):
    """
    Function to parallelize a function :)
    """
    pool = mp.Pool(num_cores)
    partial_f = partial(func, **kwargs)
    try:
        print 'starting the pool map'
        res = pool.map(partial_f, arg_list)
        pool.close()
        print 'pool map complete'
        return res
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
