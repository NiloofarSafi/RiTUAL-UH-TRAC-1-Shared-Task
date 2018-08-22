import json
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class SentimentFeatures(BaseEstimator,TransformerMixin):
    def __init__(self, mean=True):
        self._load_json(['train_dict.json', 'dev_dict.json'])
        self.flag = mean


    def _load_json(self, paths):
        self.sent_dict = {}
        for path in paths:
            with open(path) as json_data:
                d = json.load(json_data)
            json_data.close()
            for key, value in d.iteritems():
                self.sent_dict[key] = value


    def fit(self, documents, y=None):
        return self


    def transform(self, documents):

        vectors = []
        if self.flag:
            for doc in documents:
                vectors.append(self.sent_dict[doc.id]['means'])
        else:
            for doc in documents:
                vectors.append(self.sent_dict[doc.id]['stds'])
        X = np.array(vectors)
        return X

