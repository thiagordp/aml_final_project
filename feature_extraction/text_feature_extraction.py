import logging

import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.feature_extraction.text import *
from gensim.test.utils import get_tmpfile


class EmbeddingsVectorizer:

    def __init__(self, max_features=20000):
        temp = get_tmpfile("glove2word2vec.txt")
        glove2word2vec("dataset/embeddings/glove_pretrained_legal_1Bi.txt", temp)

        word_vectors = KeyedVectors.load_word2vec_format(temp, binary=False)
        self.glove_embeddings = word_vectors
        self.num_features = word_vectors.vector_size
        self.oov_random_vector = np.random.rand(self.num_features)

    def fit_transform(self, corpus):
        return self._build_doc_vector(corpus)

    def transform(self, corpus):
        return self._build_doc_vector(corpus)

    def get_feature_names_out(self):
        return ["F" + str(i) for i in range(self.num_features)]

    def _build_doc_vector(self, corpus):
        logging.info("Building GLove vectors for corpus")
        result = []
        for doc in corpus:
            tokens = doc.split()
            featureVec = np.zeros((self.num_features,), dtype="float32")
            nwords = 0

            for token in tokens:
                try:
                    token_emb = self.glove_embeddings[token]
                except:
                    token_emb = self.oov_random_vector

                featureVec = np.add(featureVec, token_emb)
                nwords += 1

            featureVec = np.divide(featureVec, nwords)
            result.append(featureVec)

        return np.array(result)


def extract_bow(corpus, method="TF", fitted_bow=None, ngram=(1, 2), max_features=5000):
    if fitted_bow:
        bow_corpus = fitted_bow.transform(corpus)
        if method != "EMB":
            bow_corpus = bow_corpus.toarray()
        return bow_corpus

    dict_method = {
        "TF": CountVectorizer(max_features=max_features, ngram_range=ngram),
        "TF-IDF": TfidfVectorizer(max_features=max_features, ngram_range=ngram),
        "EMB": EmbeddingsVectorizer(max_features=max_features)
    }

    bow_model = dict_method[method]

    corpus = bow_model.fit_transform(corpus)
    if method != "EMB":
        corpus = corpus.toarray()
    return corpus, bow_model
