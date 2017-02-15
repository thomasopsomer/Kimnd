import spacy
from nltk.tokenize import sent_tokenize
from gensim.utils import any2unicode, deaccent
import re

nlp = spacy.load("en")

re_fw_pattern = r"----[-\s]*(Original|Forwarded).*Subject:"
re_fw_regex = re.compile(re_fw_pattern)


def preprocess_mail_body(txt, nlp):
    """ """
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
        doc = nlp(sent)
        for tok in doc:
            if (not tok.is_punct and not tok.is_stop
                and not tok.like_num and not tok.is_space
                and not tok.like_url and len(tok) > 1):
                # and not any(tok.lemma_.startswith):
                bow.append(tok.lemma_)
    return bow
