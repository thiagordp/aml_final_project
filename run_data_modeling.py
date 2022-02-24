"""

"""
import logging
import os.path
import warnings
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from numpy import where
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, \
    IsolationForest
from sklearn.feature_selection import RFECV, SelectKBest, chi2, SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
import imblearn

from explainability.explain_shap import explain_shap
from feature_extraction.text_feature_extraction import extract_bow
from preprocessing.preprocess_raw_documents import raw_corpus_preprocessing
from util.constants import PATH_PLANILHA_RAW_TEXT, PATH_PLANILHA_PROC, PATH_RESULTS
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

        base_modeling(X_train_bow, X_test_bow, y_train, y_test, bow_model.get_feature_names_out(),
                      dict_results, type_modeling="text")

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
    df.sort_values(by=["Model"], ascending=True, inplace=True)

    df.to_csv(os.path.join(PATH_RESULTS, "results_text.csv"), index=False)

    output_path = os.path.join(PATH_RESULTS, "results_text.@ext")
    plot_acc_f1(df, output_path)


def modeling_w_attributes():
    # load dataset
    logging.info("Modeling using only attributes")
    logging.info("Loading and preprocessing")
    dataset_df = pd.read_csv(PATH_PLANILHA_PROC.replace(".@ext", "_2.csv"))

    dataset_df.drop(
        columns=["Número do doc", "Resultado Doc Num", "data_documento", "data_protocolo",
                 "data_doc_extr", "orgao_origem", "Conteúdo"],
        inplace=True)
    features_df = dataset_df.drop(columns=["Resultado Doc"])

    X = np.array(features_df)
    y = np.array(dataset_df["Resultado Doc"])

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

        base_modeling(X_train, X_test, y_train, y_test, feature_names, dict_results, type_modeling="attr")
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
    df.sort_values(by=["Model"], ascending=True, inplace=True)

    df.to_csv(os.path.join(PATH_RESULTS, "results_attr.csv"), index=False)

    output_path = os.path.join(PATH_RESULTS, "results_attr.@ext")
    plot_acc_f1(df, output_path)

    # Plot F1


def modeling_w_attributes_and_text():
    # load dataset
    logging.info("Modeling using only attributes and text")
    logging.info("Loading and preprocessing")
    dataset_df = pd.read_csv(PATH_PLANILHA_PROC.replace(".@ext", "_2.csv"))

    dataset_df.drop(
        columns=["Número do doc", "Resultado Doc Num", "data_documento", "data_protocolo",
                 "data_doc_extr", "orgao_origem"],
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
    cross_val_splits = 5
    skf = StratifiedKFold(n_splits=cross_val_splits, shuffle=True, random_state=142)
    print("Training/Testing using 5-fold cross-validation")
    count = 0
    for train_index, test_index in skf.split(X, y):
        logging.info("=" * 50)

        count += 1
        logging.info("Running cross-val - %d of %d" % (count, cross_val_splits))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train_bow, bow_model = extract_bow(X_train[:, 1], method="TF-IDF")
        x_test_bow = extract_bow(X_test[:, 1], fitted_bow=bow_model)
        bow_features = bow_model.get_feature_names_out()

        X_train = np.delete(X_train, 1, 1)  # Remove second column (Conteúdo)
        X_test = np.delete(X_test, 1, 1)  # Remove second column (Conteúdo)

        features_names.extend(bow_features)

        X_train = np.concatenate((X_train, x_train_bow), axis=1)
        X_test = np.concatenate((X_test, x_test_bow), axis=1)

        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        base_modeling(X_train, X_test, y_train, y_test, features_names, dict_results, features_to_select=1000,
                      type_modeling="attr_text")

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
    df.sort_values(by=["Model"], ascending=True, inplace=True)

    df.to_csv(os.path.join(PATH_RESULTS, "results_attr_text.csv"), index=False)

    output_path = os.path.join(PATH_RESULTS, "results_attr_text.@ext")
    plot_acc_f1(df, output_path)


def plot_acc_f1(df, out_path):
    """
    Código para plotar acc and F1 em um mesmo gráfico
    Obs: usado num artigo já publicado.
    :param out_path:
    :param df:
    :return:
    """
    matplotlib.rcParams['font.family'] = "FreeSerif"

    tech_names = techs = list(df["Model"])
    accs = list(df["Mean Acc"])
    accs_std = list(df["Std Acc"])
    f1_scores = list(df["Mean F1"])
    f1_scores_std = list(df["Std F1"])

    #########################################################
    # Plot R2 and RMSE in the same plot

    # plt.figure(figsize=(15, 8))
    # plt.grid()
    fig, ax1 = plt.subplots()

    fig.set_figheight(4)
    fig.set_figwidth(9)

    color = 'tab:red'
    # ax1.set_xlabel('Technique')
    ax1.set_ylabel('Accuracy', color="black", fontsize=10)
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", linestyle=":", alpha=0.8)

    ax1.set_ylim(0, 1)

    # ax1.plot(t, r2, color=color)
    x = np.arange(len(techs))
    it_techs = 0
    bar_width = 0.45

    for i_tech in range(len(techs)):
        data = accs[i_tech]
        yerr = accs_std[i_tech]

        if it_techs == 0:
            ax1.bar(x[it_techs] - (bar_width / 2), data, yerr=yerr, color=(217 / 255, 198 / 255, 176 / 255),
                    width=bar_width,
                    label="Accuracy")
        else:
            ax1.bar(x[it_techs] - (bar_width / 2), data, yerr=yerr, color=(217 / 255, 198 / 255, 176 / 255),
                    width=bar_width)

        label = "{:.1%}".format(data)

        plt.annotate(label,  # this is the text
                     (x[it_techs] - (bar_width / 2), data),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 2),  # distance from text to points (x,y)
                     color="black",
                     fontsize=10,
                     rotation=0,
                     ha='center')  # horizontal alignment can be left, right or center

        it_techs += 1
    it_techs = 0

    labels_techs = [""]

    for tech in techs:
        labels_techs.append("")
        labels_techs.append(tech)

    ax1.set_xticklabels(labels_techs, rotation=20, fontsize=12)
    ylabels = ["{:.1%}".format(label) for label in ax1.get_yticks()]
    ax1.set_yticklabels(ylabels, fontsize=12)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylim(0, 1)

    for i_tech in range(len(techs)):
        data = f1_scores[i_tech]
        yerr = f1_scores_std[i_tech]
        if it_techs == 0:
            ax2.bar(x[it_techs] + (bar_width / 2), data, yerr=yerr, color=(120 / 255, 159 / 255, 138 / 255),
                    width=bar_width,
                    label="F1-Score")
        else:
            ax2.bar(x[it_techs] + (bar_width / 2), data, yerr=yerr, color=(120 / 255, 159 / 255, 138 / 255),
                    width=bar_width)

        label = format(data, ',.2f')

        plt.annotate(label,  # this is the text
                     (x[it_techs] + (bar_width / 2), data),  # this is the point to label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 2),  # distance from text to points (x,y)
                     color="black",
                     fontsize=10,
                     rotation=0,
                     ha='center')  # horizontal alignment can be left, right or center

        it_techs += 1

    ax2.tick_params(axis='y', labelcolor="black", labelsize=12)
    # ax2.set_xticklabels(techs, rotation=30, fontsize=12)

    ylabels = [format(label, ',.2f') for label in ax1.get_yticks()]
    ax2.set_yticklabels(ylabels)
    ax2.set_ylabel('F1-Score', color="black", fontsize=10)

    labels = ["Accuracy", "F1-Score"]
    legend_elements = [
        Patch(facecolor=(217 / 255, 198 / 255, 176 / 255), label='Accuracy'),
        Patch(facecolor=(120 / 255, 159 / 255, 138 / 255), label='F1-Score'),
    ]
    ax1.legend(handles=legend_elements, fancybox=False, shadow=False, fontsize=10)

    # ax2.legend(loc='upper left', bbox_to_anchor=(0.5, 0.2), fancybox=False, shadow=False, ncol=6)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.savefig(out_path.replace("@ext", "png"), dpi=200)
    plt.savefig(out_path.replace("@ext", "svg"), dpi=200)
    plt.savefig(out_path.replace("@ext", "pdf"), dpi=200)


def model_selection(model, params, x_train, y_train):
    pass


def base_modeling(x_train, x_test, y_train, y_test, features_names, dict_results=None, features_to_select=64,
                  type_modeling=""):
    if dict_results is None:
        dict_results = dict()

    models = {
        "SVM": SVC(max_iter=500),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 32, 32), early_stopping=True, shuffle=True),
        "Naive Bayes": MultinomialNB(),
        "Adaboost": AdaBoostClassifier(n_estimators=100)
    }

    hyper_params = {
        "SVM": {
            "C": [1, 2, 5, 10, 100, 1000],
            'gamma': [0.001, 0.0001, 0.01, 0.1, 0.5, 1],
            "kernel": ["rbf", "poly", "sigmoid"],
            "coef0": [0, 0.01, 0.1, 0.5, 1]
        },
        "MLP": {
            "hidden_layer_sizes": [10, 50, 100, (32, 32), (32, 32, 32)],
            "activation": ["logistic", "relu"],
            "batch_size": [8, 16, 32, 64, 128],
            "solver": ["sgd", "adam"]
        },
        "Naive Bayes": {
            "alpha": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "fit_prior": [True, False],
        },
        "Adaboost": {
            "n_estimators": [8, 16, 32, 64, 128, 256, 512],
            "learning_rate": [0.001, 0.01,0.1, 0.5, 1],
            "algorithm": ["SAMME", "SAMME.R"],
            "base_estimator": [
                DecisionTreeClassifier(max_depth=1),
                DecisionTreeClassifier(max_depth=2),
                MLPClassifier(hidden_layer_sizes=(4,)),
                MLPClassifier(hidden_layer_sizes=(8,)),
                MLPClassifier(hidden_layer_sizes=(16,))
            ]
        }
    }

    # eu diria remoção de outliers primeiro, depois feature selection,
    # depois subsampling/oversampling, isso pra essas técnicas não influenciarem na tua seleção de features
    # lembrando que se for usar cross-validation tem que deixar under/oversampling pra depois
    logging.info("Outliers removal")
    logging.info("Datasamples for each class before remove outliers")
    model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, n_jobs=4)

    print(Counter(y_train))
    predictions = model.fit_predict(x_train, y_train)
    outliers_index = where(predictions != -1)
    x_train = x_train[outliers_index]
    y_train = y_train[outliers_index]

    predict_test = model.predict(x_test)
    outliers_index = where(predict_test != -1)
    x_test = x_test[outliers_index]
    y_test = y_test[outliers_index]

    if x_train.shape[1] >= 500:
        logging.info("Running Feature selection")
        selectFS = SelectKBest(chi2, k=100)
        x_train = selectFS.fit_transform(x_train, y_train)
        x_test = selectFS.transform(x_test)
        supports = selectFS.get_support(indices=True)
        features = pd.Series(features_names)[supports]
    else:
        features = features_names

    logging.info("Oversampling")
    logging.info("Datasamples for each class before oversampling")
    counter = Counter(y_train)
    print(Counter(y_train))
    # under = RandomUnderSampler(sampling_strategy=0.4)
    # x_train, y_train = under.fit_resample(x_train, y_train)
    over = SMOTE(sampling_strategy=0.4)
    x_train, y_train = over.fit_resample(x_train, y_train)
    logging.info("Datasamples for each class after oversampling/undersampling")
    counter = Counter(y_train)
    print(counter)


    logging.info("Training and testing models")
    count = 0
    for model_name in models.keys():
        if model_name not in dict_results.keys():
            dict_results[model_name] = {"acc": [], "f1": []}

        logging.info("Running model selection for %s" % model_name)
        model = models[model_name]
        hyper = hyper_params[model_name]

        cv = GridSearchCV(model, hyper, cv=5, verbose=3, n_jobs=8)
        cv.fit(x_train, y_train)

        logging.info("Best parameters set found on training set:\n")
        logging.info(cv.best_params_)

        model = cv.best_estimator_

        logging.info("-"*50)
        logging.info("Training %s classifier" % model_name)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        if model_name in ["Adaboost"]:

            features_imp = []
            for feat, imp in zip(features, model.feature_importances_):
                features_imp.append([feat, imp])
            df = pd.DataFrame(features_imp, columns=["Feature", "Importance"])
            df.sort_values(by=["Importance"], ascending=False, inplace=True)
            df.to_excel(os.path.join(PATH_RESULTS, "feature_imp_" + type_modeling + ".xlsx"), index=False)

        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average="macro")

        dict_results[model_name]["acc"].append(acc)
        dict_results[model_name]["f1"].append(f1)

        # logging.info("Acc: %.4f \tF1: %.4f" % (acc, f1))

    return dict_results


if __name__ == "__main__":
    setup_logging()
    logging.info("=" * 50)
    logging.info("DATA MODELLING")
    plt.rc('axes', axisbelow=True)

    modeling_w_text_only()
    # modeling_w_attributes_and_text()
    # modeling_w_attributes()
