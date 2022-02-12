"""

"""
import logging
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.feature_selection import RFECV, SelectKBest, chi2, SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier

from feature_extraction.text_feature_extraction import extract_bow
from preprocessing.preprocess_raw_documents import raw_corpus_preprocessing
from util.constants import PATH_PLANILHA_RAW_TEXT, PATH_PLANILHA_PROC
import nltk

from util.setup_logging import setup_logging

nltk.download('punkt')
warnings.filterwarnings("ignore")


def modeling_w_text_only():
    # Load dataset
    logging.info("Modeling using only raw text")
    logging.info("Loading and preprocessing")
    dataset_df = pd.read_csv(PATH_PLANILHA_RAW_TEXT.replace("@ext", "csv"))
    X = np.array(dataset_df["Conteúdo"])
    y = np.array(dataset_df["Resultado Doc"])

    dict_results = {}
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, train_size=0.7)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=142)
    print("Training/Testing using 5-fold cross-validation")
    for train_index, test_index in skf.split(X, y):
        print("=", end="")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train_bow, bow_model = extract_bow(X_train, method="TF-IDF")
        X_test_bow = extract_bow(X_test, fitted_bow=bow_model)

        scaler = StandardScaler(with_mean=False)
        X_train_bow = scaler.fit_transform(X_train_bow)
        X_test_bow = scaler.transform(X_test_bow)

        base_modeling(X_train_bow, X_test_bow, y_train, y_test, bow_model.get_feature_names_out(), dict_results)
    print("")

    data = []
    for model_name in dict_results.keys():
        accs = dict_results[model_name]["acc"]
        mean_acc = np.mean(accs)
        std_dev_acc = np.std(accs)
        f1s = dict_results[model_name]["f1"]
        mean_f1 = np.mean(f1s)
        std_dev_f1 = np.std(f1s)

        data.append([model_name, mean_acc, std_dev_acc, mean_f1, std_dev_f1])

    df = pd.DataFrame(data, columns=["Model", "Mean Acc", "Std Acc", "Mean F1", "Std F1"])

    # Plot Ac
    x, y, e = df["Model"], df["Mean Acc"], df["Std Acc"]
    plt.title("Accuracy for the models using only report text")
    plt.errorbar(x, y, e, linestyle='None', marker='^')
    plt.show()

    plt.title("F1-Score for the models using only report text")
    x, y, e = df["Model"], df["Mean F1"], df["Std F1"]
    plt.errorbar(x, y, e, linestyle='None', marker='^')
    plt.show()

    # Plot F1


def modeling_w_attributes():
    # load dataset
    logging.info("Modeling using only attributes")
    logging.info("Loading and preprocessing")
    dataset_df = pd.read_csv(PATH_PLANILHA_PROC.replace(".@ext", "_2.csv"))

    dataset_df.drop(
        columns=["Número do doc", "Resultado Doc Num", "data_documento", "ano_documento", "data_protocolo",
                 "diff_datas", "orgao_origem", "Conteúdo"],
        inplace=True)
    features_df = dataset_df.drop(columns=["Resultado Doc"])

    X = np.array(features_df)
    y = np.array(dataset_df["Resultado Doc"])
    print("Nan:", list(features_df.isnull().sum()))

    feature_names = list(features_df.columns)
    # print("Features: ", feature_names)
    dict_results = {}
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, train_size=0.7)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=142)
    print("Training/Testing using 5-fold cross-validation")
    for train_index, test_index in skf.split(X, y):
        print("=", end="")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        base_modeling(X_train, X_test, y_train, y_test, feature_names, dict_results)
    print("")

    data = []
    for model_name in dict_results.keys():
        accs = dict_results[model_name]["acc"]
        mean_acc = np.median(accs)
        std_dev_acc = np.std(accs)
        f1s = dict_results[model_name]["f1"]
        mean_f1 = np.median(f1s)
        std_dev_f1 = np.std(f1s)

        data.append([model_name, mean_acc, std_dev_acc, mean_f1, std_dev_f1])

    df = pd.DataFrame(data, columns=["Model", "Mean Acc", "Std Acc", "Mean F1", "Std F1"])

    # Plot Ac
    x, y, e = df["Model"], df["Mean Acc"], df["Std Acc"]
    plt.title("Accuracy for the models using only report text")
    plt.errorbar(x, y, e, linestyle='None', marker='^')
    plt.show()

    plt.title("F1-Score for the models using only report text")
    x, y, e = df["Model"], df["Mean F1"], df["Std F1"]
    plt.errorbar(x, y, e, linestyle='None', marker='^')
    plt.show()

    # Plot F1


def modeling_w_attributes_and_text():
    # load dataset
    logging.info("Modeling using only attributes")
    logging.info("Loading and preprocessing")
    dataset_df = pd.read_csv(PATH_PLANILHA_PROC.replace(".@ext", "_2.csv"))

    dataset_df.drop(
        columns=["Número do doc", "Resultado Doc Num", "data_documento", "ano_documento", "data_protocolo",
                 "diff_datas", "orgao_origem"],
        inplace=True)
    features_df = dataset_df.drop(columns=["Resultado Doc"])
    features_names = list(features_df.drop(columns=["Conteúdo"]).columns)

    X = np.array(features_df)
    y = np.array(dataset_df["Resultado Doc"])
    # print("Nan:", list(features_df.isnull().sum()))

    feature_names = list(features_df.columns)
    # print("Features: ", feature_names)
    dict_results = {}

    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, train_size=0.7)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=142)
    print("Training/Testing using 5-fold cross-validation")
    for train_index, test_index in skf.split(X, y):
        print("=", end="")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train_bow, bow_model = extract_bow(X_train[:, 1], method="TF-IDF")
        x_test_bow = extract_bow(X_test[:, 1], fitted_bow=bow_model)
        bow_features = bow_model.get_feature_names_out()

        X_train = np.delete(X_train, 1, 1)  # Remove second column (Conteúdo)
        X_test = np.delete(X_test, 1, 1)  # Remove second column (Conteúdo)

        features_names.extend(bow_features)
        #print("Features: ", features_names)
        print(X_train.shape)

        print(x_train_bow.shape)
        X_train = np.concatenate((X_train, x_train_bow), axis=1)
        X_test = np.concatenate((X_test, x_test_bow), axis=1)

        print(X_train.shape)
        scaler = StandardScaler(with_mean=False)
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        base_modeling(X_train, X_test, y_train, y_test, feature_names, dict_results, features_to_select=1000)
    print("")

    data = []
    for model_name in dict_results.keys():
        accs = dict_results[model_name]["acc"]
        mean_acc = np.median(accs)
        std_dev_acc = np.std(accs)
        f1s = dict_results[model_name]["f1"]
        mean_f1 = np.median(f1s)
        std_dev_f1 = np.std(f1s)

        data.append([model_name, mean_acc, std_dev_acc, mean_f1, std_dev_f1])

    df = pd.DataFrame(data, columns=["Model", "Mean Acc", "Std Acc", "Mean F1", "Std F1"])

    # Plot Ac
    x, y, e = df["Model"], df["Mean Acc"], df["Std Acc"]
    plt.title("Accuracy for the models using only report text")
    plt.errorbar(x, y, e, linestyle='None', marker='s')
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.grid("--", axis="y", which="major")
    plt.show()

    plt.title("F1-Score for the models using only report text")
    x, y, e = df["Model"], df["Mean F1"], df["Std F1"]
    plt.errorbar(x, y, e, linestyle='None', marker='s')
    plt.ylim([0, 1])
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.grid("--", axis="y", which="major")
    plt.show()


def base_modeling(x_train, x_test, y_train, y_test, features, dict_results=None, features_to_select=64):
    if dict_results is None:
        dict_results = dict()

    models = {
        "SVM": SVC(),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 32, 32)),
        "Naive Bayes": MultinomialNB(),
        "Adaboost": AdaBoostClassifier(n_estimators=100),
        "Decision Tree": DecisionTreeClassifier(max_depth=20),
        "XGBoost": XGBClassifier(n_estimators=100),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20)
    }

    # rfecv = RFECV(
    #     estimator=DecisionTreeClassifier(max_depth=20),
    #     min_features_to_select=features_to_select,
    #     n_jobs=-1,
    #     scoring="accuracy",
    #     cv=5,
    #     verbose=True
    # )

    clf = AdaBoostClassifier(n_estimators=20)
    selectFS = SelectFromModel(clf, prefit=False, max_features=200)
    # Analyze the performance according to N of features
    logging.info("Running RFE selection")
    x_train = selectFS.fit_transform(x_train, y_train)
    x_test = selectFS.transform(x_test)

    # print("Total columns: ", len(features), " Selected columns: ",
    #       len(np.array(features)[rfecv.support_]))
    # print(list(np.array(features)[rfecv.support_]))

    for model_name in models.keys():
        if model_name not in dict_results.keys():
            dict_results[model_name] = {"acc": [], "f1": []}

        # logging.info("-"*50)
        # logging.info("Training %s classifier" % model_name)
        model = models[model_name]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        # if model_name in ["XGBoost"]:
        #     logging.info("=" * 50)
        #     logging.info("%s: Feature importances (> 0.2)" % model_name)
        #     for feat, imp in zip(features, model.feature_importances_):
        #         if imp >= 0.01:
        #             logging.info("%s: %.3f" % (feat, imp))

        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average="macro")

        dict_results[model_name]["acc"].append(acc)
        dict_results[model_name]["f1"].append(f1)

        # logging.info("Acc: %.4f \tF1: %.4f" % (acc, f1))

    return dict_results


if __name__ == "__main__":
    setup_logging()
    # modeling_w_text_only()
    modeling_w_attributes_and_text()
