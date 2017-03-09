# coding: utf-8

import numpy as np
import pandas as pd
import utils
import string


def search_greetings(dataset):
    """

    """
    i = 0
    for ind, row in dataset.iterrows():
        detect_greetings(row["body"])
        i += 1
        if i == 50:
            break


def parse_firstnames(dataset):
    emails = set()
    for i, recipients in enumerate(dataset['recipients']):
        emails.update(set(recipients))

    firstnames = [mail.split('.')[0] for mail in list(emails) if
                  '.' in mail and (mail.index('.') < mail.index('@'))]

    firstnames = filter(lambda f: len(f) > 1, firstnames)

    return list(set(firstnames))


def detect_greetings(body):
    """take the body of a string as input and return a list of inputs"""
    max_length = max(100, len(body))
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
    print words
