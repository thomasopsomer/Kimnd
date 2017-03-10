# coding: utf-8

import numpy as np
import pandas as pd
import utils
import string

from data.stopwords import extendedstopwords


def search_greetings(dataset):
    """

    """
    firstnames = parse_firstnames(dataset)
    i = 0
    for ind, row in dataset.iterrows():
        detect_greetings(row["body"], firstnames)
        i += 1
        if i == 50:
            break


def parse_firstnames(dataset):
    emails = set()
    for i, recipients in enumerate(dataset['recipients']):
        emails.update(set(recipients))

    firstnames = [mail.split('.')[0] for mail in list(emails) if
                  '.' in mail and (mail.index('.') < mail.index('@'))]

    firstnames = list(set(firstnames))
    firstnames = filter(lambda f: len(f) > 1, firstnames)
    firstnames = [fn.lower for fn in firstnames]
    return firstnames


def parse_lastnames(dataset):
    emails = set()
    for i, recipients in enumerate(dataset['recipients']):
        emails.update(set(recipients))

    lastnames = [mail.split('@')[0] for mail in list(emails) if
                 '.' in mail and (mail.index('.') < mail.index('@') - 1)]
    lastnames = [mail.split('.')[1] for mail in lastnames]

    lastnames = list(set(lastnames))
    lastnames = filter(lambda f: len(f) > 1, lastnames)
    lastnames = [fn.lower for fn in lastnames]
    return lastnames


greetings_words = [
    "hi", "hey", "hello", "thank", "dear", "kiss", "mr", "madam", "miss"]


def detect_greetings(body, firstnames):
    """take the body of a string as input and return a list of inputs"""
    def tokenize(body, max_length=100):
        max_length = max(max_length, len(body))
        body = body[:max_length]
        start = 0
        end = 0
        words = []
        for i in range(max_length):
            if body[i] in string.punctuation+' ':
                words.append(body[start:end])
                start = i+1
                end = start
            elif i < max_length - 1 and body[i].islower() and body[i+1].isupper():
                end += 1
                words.append(body[start:end])
                start = i+1
                end = start
            else:
                end += 1
        words = filter(lambda f: len(f) > 1, words)
        return words
    words = tokenize(body)
    if words[0] in firstnames and not words[0] in firstnames:
        return words[0]
    print words


def parse_names_from_dataset(df):
    noms = []
    spacy.load('en')
    for doc in nlp.pipe(df["body"].str.decode('utf-8'), batch_size=10000,
                        n_threads=3):
        noms += filter(lambda m: m.ent_type_ == 'PERSON', doc)

    noms = [tok.lemma_ for tok in noms]

    for i, nom in enumerate(noms):
        if '/' in nom:
            noms[i] = nom.split('/')[0]

    noms = list(set(noms))

    for i, nom in enumerate(noms):
        n = ""
        for s in nom:
            if not s.isdigit():
                n += s
        noms[i] = n

    noms = filter(lambda n: n not in extendedstopwords, noms)

    def contain_punct(mot):
        for p in string.punctuation:
            if p in mot:
                return False
        return True

    noms = filter(contain_punct, noms)
