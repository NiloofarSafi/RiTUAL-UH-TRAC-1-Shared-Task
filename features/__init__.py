# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from . import lexical
from . import embeddings
from . import LIWC
from . import fasttext
from . import sentiment
import logging
import string
from itertools import chain
from preprocess import ark_tweet_tokenizer
from helpers.list_helpers import flattern

log = logging.getLogger(__name__)

__all__ = ['lexical', 'embeddings', 'phonetic', 'readability', 'writing_density', 'sentiments', 'get_feature',
           'create_feature', 'dumped_features', 'badwords', 'domain', 'emotions']


def preprocess(x):
    return x.replace('\n', ' ').replace('\r', '').replace('\x0C', '').lower()


def get_feature(f_name):
    """Factory to create features objects

    Parameters
    ----------
    f_name : features name

    Returns
    ----------
    features: BaseEstimator
        feture object

    """
    features_dic = dict(



        unigram=lexical.NGramTfidfVectorizer(ngram_range=(1, 1), tokenizer=ark_tweet_tokenizer, analyzer="word",
                                             lowercase=True, min_df=2),
        bunigram=lexical.NGramTfidfVectorizer(ngram_range=(1, 1), tokenizer=ark_tweet_tokenizer, analyzer="word",
                                             lowercase=True, min_df=2, binary=True),
        bigram=lexical.NGramTfidfVectorizer(ngram_range=(2, 2), tokenizer=ark_tweet_tokenizer, analyzer="word",
                                            lowercase=True, min_df=5),
        trigram=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), tokenizer=ark_tweet_tokenizer, analyzer="word",
                                             lowercase=True, min_df=5),

        #
        # #char ngram
        char_tri=lexical.NGramTfidfVectorizer(ngram_range=(3, 3), analyzer="char",
                                              lowercase=True, min_df=5),
        char_4_gram=lexical.NGramTfidfVectorizer(ngram_range=(4, 4), analyzer="char", lowercase=True, min_df=5),

        char_5_gram=lexical.NGramTfidfVectorizer(ngram_range=(5, 5), analyzer="char", lowercase=True, min_df=5),


        # #skip gram
        two_skip_3_grams=lexical.KSkipNgramsVectorizer(k=2, tokenizer=ark_tweet_tokenizer, ngram=3, lowercase=True),
        two_skip_2_grams=lexical.KSkipNgramsVectorizer(k=2, tokenizer=ark_tweet_tokenizer, ngram=2, lowercase=True),


        google_word_emb=embeddings.Word2VecFeatures(tokenizer=ark_tweet_tokenizer, analyzer="word",
                                                    lowercase=True,
                                                    model_name='/Users/niloofar/PycharmProjects/eRisk_experiments/resources/GoogleNews-vectors-negative300.bin'),

        liwc = LIWC.LIWCFeature(),

        fast_text = fasttext.FasttextFeatures(),

        sent_mean = sentiment.SentimentFeatures(mean=True),
        sent_std = sentiment.SentimentFeatures(mean=False)
    )

    return features_dic[f_name]


def create_feature(feature_names):
    """Utility function to create features object

    Parameters
    -----------
    feature_names : features name or list of features names


    Returns
    --------
    a tuple of (feature_name, features object)
       lst features names are joined by -
       features object is the union of all features in the lst

    """

    def feature_creater(f_names):
        try:
            if isinstance(f_names, list):
                return "-".join(f_names), FeatureUnion([(f, get_feature(f)) for f in f_names])
            else:

                return f_names, get_feature(f_names)
        except Exception as e:
            log.error("Error:: {} ".format(e))
            raise ValueError('Error in function ')

    if isinstance(feature_names, list):
        return  "-".join(flattern(feature_names)), FeatureUnion([(feature_creater(f)) for f in feature_names])

    else:
        return feature_creater(feature_names)
