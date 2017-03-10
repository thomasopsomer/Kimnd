#!/usr/bin/env python
# -*- coding: utf-8 -*-
import spacy
from spacy.tokenizer import Tokenizer
import json
from nltk.tokenize import sent_tokenize
from gensim.utils import any2unicode, deaccent
import re

try:
    import regex
    REGEX = True
    lower_upper_pat = regex.compile("(?<=[a-z])(?=[A-Z])",
                                    flags=regex.VERSION1)
    number_letter_pat = regex.compile("(?<=[0-9])(?=[a-zA-Z])",
                                      flags=regex.VERSION1)
except ImportError:
    REGEX = False


re_fw_pattern = r"----[-\s]*(Original|Forwarded).*Subject:"
re_fw_regex = re.compile(re_fw_pattern)


infix_entries = \
    """
    \.\.\.
    (?<=[a-z0-9])\.(?=[a-zA-Z0-9\[\]\_\-\*])
    (?<=[a-zA-Z0-9])--(?=[a-zA-z0-9])
    (?<=[a-zA-Z0-9])-(?=[a-zA-Z0-9])
    (?<=[a-zA-Z0-9])\?(?=[a-zA-Z0-9])
    (?<=[a-zA-Z0-9])\!(?=[a-zA-Z0-9])
    (?<=[A-Za-z0-9]),(?=[A-Za-z\[])
    (?<=[0-9])-(?=[0-9])
    """
infix_re = spacy.util.compile_infix_regex(infix_entries.split())


def create_tokenizer(nlp):
    """ """
    target_name, target_version = spacy.util.split_data_name("en")
    data_path = spacy.util.get_data_path()
    path = spacy.util.match_best_version(
        target_name, target_version, data_path)
    #
    # solve pb with open file encoding
    with (path / 'tokenizer' / 'specials.json').open(encoding='utf8') as file_:
        rules = json.load(file_)
    with (path / 'tokenizer' / 'prefix.txt').open(encoding="utf-8") as file_:
        entries = file_.read().split('\n')
    prefix_search = spacy.util.compile_prefix_regex(entries).search
    with (path / 'tokenizer' / 'suffix.txt').open(encoding="utf-8") as file_:
        entries = file_.read().split('\n')
    suffix_search = spacy.util.compile_suffix_regex(entries).search
    #
    return Tokenizer.load(path, nlp.vocab,
                          rules=rules,
                          prefix_search=prefix_search,
                          suffix_search=suffix_search,
                          infix_finditer=infix_re.finditer)


def get_custom_spacy(parser=False, entity=False):
    """ """
    nlp = spacy.load("en", parser=parser, entity=entity)
    tokenizer = create_tokenizer(nlp)
    nlp.tokenizer = tokenizer
    return nlp


not_in_list = ["_", "--", "*", "/", ":", "=", "(", "]", ")", "#", "|", "@", "+"]


def bow_mail_body(txt, nlp):
    """
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
            if (tok.lemma_ and
                not tok.is_punct and not tok.is_stop and
                not tok.like_num and not tok.is_space and
                not tok.like_url and len(tok) > 1 and
                not any((x in tok.orth_ for x in not_in_list))):
                if tok.orth_.startswith("-") or tok.orth_.endswith("-"):
                    bow.append(tok.lemma_.replace("-", ""))
                else:
                    bow.append(tok.lemma_)
    return bow


def sent_mail_body(txt, nlp):
    """
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
    sents = []
    for sent in sentences:
        s = []
        if REGEX:
            sent = " ".join(lower_upper_pat.split(sent))
            sent = " ".join(number_letter_pat.split(sent))
        doc = nlp(sent, parse=False, entity=False)
        for tok in doc:
            if (tok.lemma_ and
                not tok.is_punct and not tok.is_stop and
                not tok.like_num and not tok.is_space and
                not tok.like_url and len(tok) > 1 and
                not any((x in tok.orth_ for x in not_in_list))
               ):
                if tok.orth_.startswith("-") or tok.orth_.endswith("-"):
                    s.append(tok.lemma_.replace("-", ""))
                else:
                    s.append(tok.lemma_)
        sents.append(s)
    return sents


def preprocess_txt(raw_txt):
    """ """
    txt = deaccent(any2unicode(raw_txt))
    txt = "\n".join(re_fw_regex.split(txt))
    txt = txt.replace(">", " ")
    txt = " ".join(lower_upper_pat.split(txt))
    txt = " ".join(number_letter_pat.split(txt))
    return res


def extract_nlp(doc):
    """ """
    for sent in doc.sents


if __name__ == '__main__':
    #
    # nlp = get_custom_spacy()
    # s = u"Hi JasonTed let me know of the situation.I'm going home."
    # s = " ".join(pat.split(s))
    # d = nlp(s)
    # for x in d: print x.orth_, x.lemma_
    pass
