import numpy as np
import pandas as pd
import spacy
import keras
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam

from RecPredicter import RecPredicter


def count_entity_sentiment(nlp, texts):
    '''Compute the net document sentiment for each entity in the texts.'''
    entity_sentiments = collections.Counter(float)
    for doc in nlp.pipe(texts, batch_size=1000, n_threads=4):
        for ent in doc.ents:
            entity_sentiments[ent.text] += doc.sentiment
    return entity_sentiments


def load_nlp(lstm_path, lang_id='en'):
    def create_pipeline(nlp):
        return [nlp.tagger, nlp.entity, RecPredicter.load(lstm_path, nlp)]
    return spacy.load(lang_id, create_pipeline=create_pipeline)


def train(train_texts, train_labels, dev_texts, dev_labels,
          lstm_shape, lstm_settings, batch_size=100,
          nb_epoch=5):
    nlp = spacy.load('en', parser=False, tagger=False, entity=False)
    embeddings = get_embeddings(nlp.vocab)
    model = compile_lstm(embeddings, lstm_shape, lstm_settings)
    train_X = get_features(nlp.pipe(train_texts))
    dev_X = get_features(nlp.pipe(dev_texts))
    model.fit(train_X, train_labels, validation_data=(dev_X, dev_labels),
              nb_epoch=nb_epoch, batch_size=batch_size)
    return model


def compile_lstm(embeddings, shape, settings):
    model = Sequential()
    model.add(
        Embedding(
            embeddings.shape[1],
            embeddings.shape[0],
            input_length=shape['max_length'],
            trainable=False,
            weights=[embeddings]
        )
    )
    model.add(Bidirectional(LSTM(shape['nr_hidden'])))
    model.add(Dropout(settings['dropout']))
    model.add(Dense(shape['nr_class'], activation='sigmoid'))
    model.compile(optimizer=Adam(lr=settings['lr']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def get_embeddings(vocab):
    max_rank = max(lex.rank for lex in vocab if lex.has_vector)
    vectors = np.ndarray((max_rank + 1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors


def get_features(docs, max_length):
    Xs = np.zeros(len(list(docs)), max_length, dtype='int32')
    for i, doc in enumerate(docs):
        for j, token in enumerate(doc[:max_length]):
            Xs[i, j] = token.rank if token.has_vector else 0
    return Xs


if __name__ == '__main__':

    data = pd.read_csv("info2.csv")
    train_indices = np.zeros(data.shape[0], dtype='bool')

    train_indices[np.random.choice(data.shape[0],
                                   size=int(data.shape[0] * 0.95),
                                   replace=False)] = True

    test_indices = 1 - train_indices * 1
    test_indices = test_indices.astype('bool')
    train_text = data.loc[train_indices]['body']
    train_labels = data.loc[train_indices]['recipients']
    dev_text = data.loc[test_indices]['body']
    dev_labels = data.loc[test_indices]['recipients']

    # train_labels = []
    # test_labels = []
    # for i, row in enumerate(data[train_indices]):
    #     train_labels.append([int(s) for s in row["recipients"].split(',') if s.isdigit()])
    #
    # for i, row in enumerate(data[test_indices]["recipients"]):
    #     test_labels.append([int(s) for s in row.split(',') if s.isdigit()])
