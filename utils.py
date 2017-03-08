#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import string
from os import path

import pandas as pd
import cPickle as pkl
import numpy as np
from spacy import en
import spacy

from nltk.tokenize import sent_tokenize
from gensim.utils import any2unicode, deaccent

# utils for loading and preprocessing dataset

# Parallelization
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

# To clean recipients
def clean_recipients(row):
    recipients = [recipient for recipient in row if "@" in recipient]
    return recipients


extendedstopwords = ("a", "about", "above", "across", "after", "MIME Version", "forwarded", "again", "against", "all", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "an", "and", "another", "any", "anybody", "anyone", "anything", "anywhere", "are", "area", "areas", "aren't", "around", "as", "ask", "asked", "asking", "asks", "at", "away", "b", "back", "backed", "backing", "backs", "be", "became", "because", "become", "becomes", "been", "before", "began", "behind", "being", "beings", "below", "best", "better", "between", "big", "both", "but", "by", "c", "came", "can", "cannot", "can't", "case", "cases", "certain", "certainly", "clear", "clearly", "come", "could", "couldn't", "d", "did", "didn't", "differ", "different", "differently", "do", "does", "doesn't", "doing", "done", "don't", "down", "downed", "downing", "downs", "during", "e", "each", "early", "either", "end", "ended", "ending", "ends", "enough", "even", "evenly", "ever", "every", "everybody", "everyone", "everything", "everywhere", "f", "face", "faces", "fact", "facts", "far", "felt", "few", "find", "finds", "first", "for", "four", "from", "full", "fully", "further", "furthered", "furthering", "furthers", "g", "gave", "general", "generally", "get", "gets", "give", "given", "gives", "go", "going", "good", "goods", "got", "great", "greater", "greatest", "group", "grouped", "grouping", "groups", "h", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "her", "here", "here's", "hers", "herself", "he's", "high", "higher", "highest", "him", "himself", "his", "how", "however", "how's", "i", "i'd", "if", "i'll", "i'm", "important", "in", "interest", "interested", "interesting", "interests", "into", "is", "isn't", "it", "its", "it's", "itself", "i've", "j", "just", "k", "keep", "keeps", "kind", "knew", "know", "known", "knows", "l", "large", "largely", "last", "later", "latest", "least", "less", "let", "lets", "let's", "like", "likely", "long", "longer", "longest", "m", "made", "make", "making", "man", "many", "may", "me", "member", "members", "men", "might", "more", "most", "mostly", "mr", "mrs", "much", "must", "mustn't", "my", "myself", "n", "necessary", "need", "needed", "needing", "needs", "never", "new", "newer", "newest", "next", "no", "nobody", "non", "noone", "nor", "not", "nothing", "now", "nowhere", "number", "numbers", "o", "of", "off", "often", "old", "older", "oldest", "on", "once", "one", "only", "open", "opened", "opening", "opens", "or", "order", "ordered", "ordering", "orders", "other", "others", "ought", "our", "ours", "ourselves", "out", "over", "own", "p", "part", "parted", "parting", "parts", "per", "perhaps", "place", "places", "point", "pointed", "pointing", "points", "possible", "present", "presented", "presenting", "presents", "problem", "problems", "put", "puts", "q", "quite", "r", "rather", "really", "right", "room", "rooms", "s", "said", "same", "saw", "say", "says", "second", "seconds", "see", "seem", "seemed", "seeming", "seems", "sees", "several", "shall", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "show", "showed", "showing", "shows", "side", "sides", "since", "small", "smaller", "smallest", "so", "some", "somebody", "someone", "something", "somewhere", "state", "states", "still", "such", "sure", "t", "take", "taken", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "therefore", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "thing", "things", "think", "thinks", "this", "those", "though", "thought", "thoughts", "three", "through", "thus", "to", "today", "together", "too", "took", "toward", "turn", "turned", "turning", "turns", "two", "u", "under", "until", "up", "upon", "us", "use", "used", "uses", "v", "very", "w", "want", "wanted", "wanting", "wants", "was", "wasn't", "way", "ways", "we", "we'd", "well", "we'll", "wells", "went", "were", "we're", "weren't", "we've", "what", "what's", "when", "when's", "where", "where's", "whether", "which", "while", "who", "whole", "whom", "who's", "whose", "why", "why's", "will", "with", "within", "without", "won't", "work", "worked", "working", "works", "would", "wouldn't", "x", "y", "year", "years", "yes", "yet", "you", "you'd", "you'll", "young", "younger", "youngest", "your", "you're", "yours", "yourself", "yourselves", "you've", "z")


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
        # clean recipients
        set_df["recipients"] = set_df["recipients"].apply(clean_recipients)
        #
        if flat:
            set_df = flatmap(set_df, "recipients", "recipient", np.string0)
    #
    set_df.date = pd.to_datetime(set_df.date)
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
re2 = re.compile('(.+)@(.+)')  # Remove emails
re3 = re.compile('\s(-----)(.*?)(-----)\s', re.DOTALL)
re4 = re.compile('''\s(\*\*\*\*\*)(.*?)(\*\*\*\*\*)\s''', re.DOTALL)
re5 = re.compile('\s(_____)(.*?)(_____)\s', re.DOTALL)
re6 = re.compile('\n( )*-.*')
re7 = re.compile('\n( )*\d.*')
re8 = re.compile(
    '(\n( )*[\w]+($|( )*\n))|(\n( )*(\w)+(\s)+(\w)+(( )*\n)|$)|(\n( )*(\w)+(\s)+(\w)+(\s)+(\w)+(( )*\n)|$)')
re9 = re.compile('.*orwarded.*')
re10 = re.compile(
    'From.*|Sent.*|cc.*|Subject.*|Embedded.*|http.*|\w+\.\w+|.*\d\d/\d\d/\d\d\d\d.*')
re11 = re.compile(' [\d:;,.]+ ')


# Replace punctuation in words by spaces
def replace_punct(s):
    for c in string.punctuation:
        s = s.replace(c, " ")
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
    # remove punctuation

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
                tok.lemma_ not in extendedstopwords and
                not tok.like_num and not tok.is_space and
                not tok.like_url and len(tok) > 1 and
                "**" not in tok.orth_ and
                    not (tok.orth_.startswith("_")) and
                    not (tok.orth_.startswith("-"))):
                bow.append(tok.lemma_)
    return bow


def preprocess_bodies(dataset, type="test"):
    pickle_path = "preprocessed_data_{:s}.pkl".format(type)
    if path.exists(pickle_path):
        texts = pkl.load(open(pickle_path, "rb"))
    else:
        texts = dataset["body"]
        texts = texts.str.decode('utf-8')
        print "Loading Spacy"
        nlp = spacy.load('en', parser=False)

        print "Preprocessing mails"
        # texts = parallelize_dataframe(texts, preprocess_mail_body, num_cores=4,
        #                               nlp=nlp)
        texts = texts.apply(preprocess_mail_body, args=(nlp,))

        texts = list(texts)
        with open(pickle_path, "w") as f:
            pkl.dump(texts, f)
    dataset["tokens"] = pd.Series(texts)
    return dataset
