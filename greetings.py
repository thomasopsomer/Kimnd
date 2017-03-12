# coding: utf-8

import numpy as np
import pandas as pd

import string
import operator

from collections import Counter

from data.stopwords import extendedstopwords
import utils
from spacy_utils import get_custom_spacy

extendedstopwords += ("2000", "product", "call", "guy", "enron", "businesses",
                      "date", "stock", "chance", "day", "information", "weeks",
                      "forward", "track", "book", "little", "com", "experience",
                      "quick")


def greeting_value(body, recipient, greets, names):
    """
    for a message body and a recipient, return 1 if the message contains an
    appropriate greeting -1 if the message contains a greeting for somebody
    else and 0 otherwise
    """
    greet = detect_greetings(body, names)

    rec_names = recipient.split('@')[0]
    if '.' in recipient:
        rec_names = rec_names.split(".")
    else:
        rec_names = [rec_names]
    rec_names = map(lambda n: (n, 1), rec_names)

    # a function that create a function that score the best
    def filtre(threshold):
        return lambda word: (word[0] in greet) and (word[1] > threshold)

    if len(greet) == 0:
        return 0.
    # import pdb; pdb.set_trace()
    if len(filter(filtre(0.5), rec_names + greets[recipient])) > 0:
        return 1
    else:
        return -1


def search_greetings(dataset, threshold=0.2):
    """
    we create a dictionary where we list the names of all 'greetings' used for
    a recipient
    """
    firstnames = parse_firstnames(dataset)
    lastnames = parse_lastnames(dataset)
    names = firstnames + lastnames
    # nlp = get_custom_spacy()
    i = 0
    greets = {}
    for ind, row in dataset.iterrows():
        # greet = utils.extract_names(row["body"], nlp)
        greet = detect_greetings(row["body"], names)
        for rec in row["recipients"]:
            if rec not in greets:
                greets[rec] = {}
            else:
                cnt = Counter(greet)
                for gr in cnt.keys():
                    if gr not in greets[rec]:
                        greets[rec][gr] = 0
                    greets[rec][gr] += float(cnt[gr]) / len(row["recipients"])
    for rec in greets.keys():
        greets[rec] = sorted(
            greets[rec].items(), key=operator.itemgetter(1), reverse=True)

    # filter extremes
    for rec in greets.keys():
        greets[rec] = filter(lambda w: w[1] > threshold, greets[rec])


    return greets


def parse_firstnames(dataset):
    emails = set()
    for i, recipients in enumerate(dataset['recipients']):
        emails.update(set(recipients))
    # only enron mails

    firstnames = [mail.split('.')[0] for mail in list(emails) if
                  '.' in mail and (mail.index('.') < mail.index('@')) and
                  mail.split('@')[1] == 'enron.com']

    firstnames = list(set(firstnames))
    firstnames = filter(lambda f: len(f) > 1, firstnames)
    firstnames = filter(lambda n: n not in extendedstopwords, firstnames)
    firstnames = [fn.lower() for fn in firstnames]
    return firstnames


def parse_lastnames(dataset):
    emails = set()
    for i, recipients in enumerate(dataset['recipients']):
        emails.update(set(recipients))

    lastnames = [mail.split('@')[0] for mail in list(emails) if
                 '.' in mail and (mail.index('.') < mail.index('@') - 1) and
                 mail.split('@')[1] == 'enron.com']
    lastnames = [mail.split('.')[1] for mail in lastnames]

    lastnames = list(set(lastnames))
    lastnames = filter(lambda f: len(f) > 1, lastnames)
    lastnames = filter(lambda n: n not in extendedstopwords, lastnames)
    lastnames = [fn.lower() for fn in lastnames]
    return lastnames


greetings_words = [
    "hi", "hey", "hello", "thank", "dear", "kiss", "mr", "madam", "miss"]


def detect_greetings(body, names):
    """take the body of a string as input and return a list of inputs"""
    def tokenize(body, max_length=100):
        max_length = min(max_length, len(body))
        body = body[:max_length]
        start = 0
        end = 0
        words = []
        for i in range(max_length):
            if body[i] in string.punctuation+' ':
                words.append(body[start:end])
                start = i+1
                end = start
            elif i < max_length - 2 and body[i].islower() and body[i+1].isupper():
                end += 1
                words.append(body[start:end])
                start = i+1
                end = start
            else:
                end += 1
        words = filter(lambda f: len(f) > 1, words)
        return words
    words = tokenize(body)
    greeting_names = filter(lambda n: n.lower() in names, words)
    return greeting_names


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


if __name__ == '__main__':
    from utils import load_dataset
    df = load_dataset()
    greets = search_greetings(df)
    names = parse_lastnames(df) + parse_firstnames(df)
    i = 0
    for id, row in df.iterrows():
        i += 1
        if i == 20:
            break
        print "\t", greeting_value(row["body"], row["recipients"][0], greets, names)
        print row["body"]
        print row["recipients"][0]
        print "\n"
