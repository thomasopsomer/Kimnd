#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

from nltk.tokenize import sent_tokenize
from gensim.utils import any2unicode, deaccent
import re
from spacy import en
import string
import numpy as np

# utils for loading and preprocessing dataset

# To clean recipients
def clean_recipients(row):
    recipients = [recipient for recipient in row if "@" in recipient]
    return recipients


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


def load_dataset(dataset_path, mail_path, train=True, flat=False):
    """
    Load and preprocess the dataset
    Explode the mail ids (mids)
    Merge with content of email to get the body
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
        set_df.recipients = set_df.recipients.str.split()
    #
    set_df.date = pd.to_datetime(set_df.date)
    # clean recipients
    set_df["recipients"] = set_df["recipients"].apply(clean_recipients)
    #
    if flat:
        set_df = flatmap(set_df, "recipients", "recipient", np.string0)
    return set_df


# Preprocessing of email content to extract Features and Cleaned text
def is_forward(txt):
    if "-----Original Message-----" in txt:
        return True
    else:
        return False


# re_fw_pattern = r"----[-\s]*(Original|Forwarded).*Subject:"
# re_fw_regex = re.compile(re_fw_pattern)

# some dude's regexes
re0 = re.compile('>')
re1 = re.compile('(Message-ID(.*?\n)*X-FileName.*?\n)|'
                 '(To:(.*?\n)*?Subject.*?\n)|'
                 '(< (Message-ID(.*?\n)*.*?X-FileName.*?\n))')
re2 = re.compile('(.+)@(.+)') # Remove emails
re3 = re.compile('\s(-----)(.*?)(-----)\s', re.DOTALL)
re4 = re.compile('''\s(\*\*\*\*\*)(.*?)(\*\*\*\*\*)\s''', re.DOTALL)
re5 = re.compile('\s(_____)(.*?)(_____)\s', re.DOTALL)
re6 = re.compile('\n( )*-.*')
re7 = re.compile('\n( )*\d.*')
re8 = re.compile('(\n( )*[\w]+($|( )*\n))|(\n( )*(\w)+(\s)+(\w)+(( )*\n)|$)|(\n( )*(\w)+(\s)+(\w)+(\s)+(\w)+(( )*\n)|$)')
re9 = re.compile('.*orwarded.*')
re10 = re.compile('From.*|Sent.*|cc.*|Subject.*|Embedded.*|http.*|\w+\.\w+|.*\d\d/\d\d/\d\d\d\d.*')
re11 = re.compile(' [\d:;,.]+ ')


# Replace punctuation in words by spaces
def replace_punct(s):
    for c in string.punctuation:
        s = s.replace(c, "")
    return s


def preprocess_mail_body(txt, nlp):
    """
    args:
        - txt: raw text
        - nlp: a spacy engine
    """
    # to unicode & get rid of accent
    txt = deaccent(any2unicode(txt))
    # split according to reply forward (get rid of "entête")

    # txt = "\n".join(re_fw_regex.split(txt))
    # txt = txt.replace(">", " ")
    txt = re.sub(re0, ' ', txt)
    txt = re.sub(re1, ' ', txt)
    txt = re.sub(re2, ' ', txt)
    txt = re.sub(re3, ' ', txt)
    txt = re.sub(re4, ' ', txt)
    txt = re.sub(re5, ' ', txt)
    txt = re.sub(re6, ' ', txt)
    txt = re.sub(re7, ' ', txt)
    txt = re.sub(re8, ' ', txt)
    txt = re.sub(re9, ' ', txt)
    txt = re.sub(re10, ' ', txt)
    txt = re.sub(re11, ' ', txt)

    # remove punctuation
    txt = replace_punct(txt)

    # parenthesis
    txt = txt.replace(')', ' ').replace('(', ' ').replace('"', '')

    # remove url
    url_exp = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    txt = re.sub(url_exp, '', txt)

    # split sentences
    sentences = sent_tokenize(txt)
    # tokenize + lemmatize + filter ?
    bow = []
    for sent in sentences:
        doc = nlp(sent, parse=False, entity=False)
        for tok in doc:
            if (not tok.is_punct and tok.lemma_ not in en.STOP_WORDS and
                not tok.like_num and not tok.is_space and
                not tok.like_url and len(tok) > 1 and
                "**" not in tok.orth_ and
                    not (tok.orth_.startswith("_")) and
                    not (tok.orth_.startswith("-"))):
                bow.append(tok.lemma_)
    return bow
