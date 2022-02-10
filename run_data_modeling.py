"""

"""
import logging

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB

from feature_extraction.text_feature_extraction import extract_bow
from preprocessing.preprocess_raw_documents import raw_corpus_preprocessing
from util.constants import PATH_PLANILHA_RAW_TEXT
import nltk

from util.setup_logging import setup_logging

nltk.download('punkt')


def modeling_w_text_only():
    # Load dataset
    logging.info("Modeling using only raw text")
    logging.info("Loading and preprocessing")
    dataset_df = pd.read_csv(PATH_PLANILHA_RAW_TEXT.replace("@ext", "csv"))
    X = list(dataset_df["Conteúdo"])
    y = list(dataset_df["Resultado Doc"])

    # Pre-processing
    X = raw_corpus_preprocessing(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, train_size=0.7)

    X_train_bow, bow_model = extract_bow(X_train, method="TF")
    X_test_bow = extract_bow(X_test, fitted_bow=bow_model)

    base_modeling(X_train_bow, X_test_bow, y_train, y_test, bow_model.get_feature_names_out())


def modeling_w_attributes():
    # load dataset

    pass


def base_modeling(x_train, x_test, y_train, y_test, features):
    models = {
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, ccp_alpha=),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 32, 32)),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(max_depth=10, n_jobs=4, n_estimators=50)
    }

    for model_name in models.keys():
        print("Training %s classifier" % model_name)
        model = models[model_name]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        if model_name in ["Decision Tree", "Random Forest"]:

            for feat, imp in zip(features, model.feature_importances_):
                if imp >= 0.01:
                    print(feat, round(imp, 3))

        print(classification_report(y_test, y_pred))

    # Train Decision tree

    pass


if __name__ == "__main__":
    setup_logging()
    modeling_w_text_only()
