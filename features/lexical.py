# -*- coding: utf-8 -*-
from __future__ import division, print_function
import string
import re
import itertools
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np




__all__ = ['NGramTfidfVectorizer','CategoricalCharNgramsVectorizer','KSkipNgramsVectorizer']


class NGramTfidfVectorizer(TfidfVectorizer):
    """Convert a collection of  documents objects to a matrix of TF-IDF features.

      Refer to super class documentation for further information
    """

    def build_analyzer(self):
        """Overrides the super class method

        Parameter
        ----------
        self

        Returns
        ----------
        analyzer : function
            extract content from document object and then applies analyzer

        """
        analyzer = super(TfidfVectorizer,
                         self).build_analyzer()
        return lambda doc: (w for w in analyzer(doc.content))

class KSkipNgramsVectorizer(TfidfVectorizer):
    """ k skip n gram Feature estimator
    Refer to http://homepages.inf.ed.ac.uk/ballison/pdf/lrec_skipgrams.pdf

    """

    def __init__(self, k=1, ngram=2, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.k = k
        self.ngram = ngram
        super(KSkipNgramsVectorizer, self).__init__(input=input, encoding=encoding, decode_error=decode_error,
                                                    strip_accents=strip_accents, lowercase=lowercase,
                                                    preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
                                                    stop_words=stop_words, token_pattern=token_pattern,
                                                    ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                                                    max_features=max_features, vocabulary=vocabulary, binary=binary,
                                                    dtype=dtype)

    def _skip_grams_sentence(self, sentence, stop_words=None):
        tokens = sentence
        if stop_words is not None:
            tokens = [w for w in sentence if w not in stop_words]

        k, ngram = self.k, self.ngram
        original_tokens = tokens
        n_original_tokens = len(original_tokens)

        skip_grams = []
        for i in range(n_original_tokens - ngram + 1):
            for x in itertools.combinations(original_tokens[i:i + k + ngram], ngram):
                skip_grams.append(x)
        return [" ".join(skip_gram) for skip_gram in sorted(set(skip_grams))]

    def _k_skip_ngrams(self, text_document, stop_words):
        tokenize = super(TfidfVectorizer, self).build_tokenizer()
        tokens = []
        for sentence in nltk.sent_tokenize(text_document):
            tokens += self._skip_grams_sentence(tokenize(sentence), stop_words)
        return tokens

    def build_analyzer(self):
        stop_words = super(TfidfVectorizer, self).get_stop_words()
        preprocess = super(TfidfVectorizer, self).build_preprocessor()
        return lambda doc: self._k_skip_ngrams(preprocess(self.decode(doc.content)), stop_words)
