from sklearn.feature_extraction.text import *


def extract_bow(corpus, method="TF", fitted_bow=None, ngram=(1,1)):
    if fitted_bow:
        return fitted_bow.transform(corpus)

    dict_method = {
        "TF": CountVectorizer(max_features=1000, ngram_range=ngram),
        "TF-IDF": TfidfVectorizer(max_features=1000, ngram_range=ngram),
    }

    bow_model = dict_method[method]
    corpus = bow_model.fit_transform(corpus)
    return corpus, bow_model
