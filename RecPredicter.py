import spacy
import numpy as np
import cytoolz


class RecPredicter(object):
    @classmethod
    def load(cls, path, nlp):
        with (path / 'config.json').open() as file_:
            model = model_from_json(file_.read())
        with (path / 'model').open('rb') as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model)

    def __init__(self, model):
        self._model = model

    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_sentiment(doc, y)

    def pipe(self, docs, batch_size=1000, n_threads=2):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            Xs = get_features(minibatch)
            ys = self._model.predict(Xs)
            for i, doc in enumerate(minibatch):
                doc.sentiment = ys[i]

    def set_sentiment(self, doc, y):
        # doc.sentiment = float(y[0])
        # Sentiment has a native slot for a single float.
        # For arbitrary data storage, there's:
        doc.user_data['my_data'] = y
