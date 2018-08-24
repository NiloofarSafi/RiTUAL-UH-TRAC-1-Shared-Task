from __future__ import division, print_function
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
import csv
from preprocess import ark_tweet_tokenizer
import sys
reload(sys)
sys.setdefaultencoding('UTF8')

__all__ = ['Word2VecFeatures']

class GenderFeatures(BaseEstimator,TransformerMixin):

    def __init__(self):
        self._load_gender_lexicon('/Users/niloofar/Downloads/emnlp2014_ageGenderLexica/emnlp14gender.csv')
        self._load_age_lexicon('/Users/niloofar/Downloads/emnlp2014_ageGenderLexica/emnlp14age.csv')

    def _load_gender_lexicon(self, file):
        with open(file, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            self.gender_dict = {rows['term']: float(rows['weight']) for rows in reader}


    def _load_age_lexicon(self, file):
        with open(file, mode='r') as csvfile:
            reader = csv.DictReader(csvfile)
            self.age_dict = {rows['term']: float(rows['weight']) for rows in reader}


    def get_feature_names(self):
        features = ["Feature_Gender", "binary_gender"]
        return np.array(features)


    def fit(self, documents, y=None):
        return self


    def extract_features(self, tokens, dict):
        doc_len = len(set(tokens))

        sum = 0
        for token in tokens:
            if token.lower() in dict:
                sum += dict[token.lower()]*(tokens.count(token)/doc_len)

        return sum + dict['_intercept']


    def transform(self, documents):

        features = []

        for doc in documents:
            tokens = ark_tweet_tokenizer(doc.content.lower())
            gender = self.extract_features(tokens,self.gender_dict)
            age = self.extract_features(tokens,self.age_dict)
            if gender < 0:
                g = 0
            else:
                g = 1
            features.append([gender, g])
        X = np.array(features)
        return X