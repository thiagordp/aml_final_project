"""

"""
import warnings

from sklearn.metrics import confusion_matrix

warnings.simplefilter("ignore")

import logging
import os.path
import warnings
from collections import Counter

import matplotlib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from numpy import where
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, \
    IsolationForest
from sklearn.feature_selection import RFECV, SelectKBest, chi2, SelectFromModel, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
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


def modeling_w_text_only(representation):
    # Load dataset
    logging.info("=" * 50)
    logging.info("Modeling using only raw text:" + representation)
    logging.info("Loading and preprocessing")
    dataset_df = pd.read_csv(PATH_PLANILHA_RAW_TEXT.replace("@ext", "csv"))
    X = np.array(dataset_df["Conteúdo"])
    y = np.array(dataset_df["Resultado Doc"])
    print(Counter(y))
    dict_results = {}

    # print("Training/Testing using 5-fold cross-validation")
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=142,
                                                                train_size=0.8)

    selected_models = {}

    # Train set is used to grid search
    # Test set is used to final metric

    X_train_bow, bow_model = extract_bow(X_train_val, method=representation, ngram=(1, 2))
    X_test_bow = extract_bow(X_test, fitted_bow=bow_model, method=representation)

    scaler = StandardScaler(with_mean=False)
    X_train_bow = scaler.fit_transform(X_train_bow)
    X_test_bow = scaler.transform(X_test_bow)
    run_model_selection = True

    base_modeling(X_train_bow, X_test_bow, y_train_val, y_test, bow_model.get_feature_names_out(),
                  dict_results, type_modeling="text_" + representation.lower(), run_model_selection=run_model_selection,
                  sel_models=selected_models)

    print("")
    logging.info("Saving metrics and results")

    data = []
    for model_name in dict_results.keys():
        accs = dict_results[model_name]["acc"]
        mean_acc = np.mean(accs)
        std_dev_acc = np.std(accs)
        f1s = dict_results[model_name]["f1"]
        mean_f1 = np.mean(f1s)
        std_dev_f1 = np.std(f1s)
        model = dict_results[model_name]["model"]
        cm = dict_results[model_name]["cm"]
        data.append([model_name, str(model), mean_acc, std_dev_acc, mean_f1, std_dev_f1, cm])

    df = pd.DataFrame(data,
                      columns=["Model", "Model Details", "Mean Acc", "Std Acc", "Mean F1", "Std F1", "Conf Matrix"])
    df.sort_values(by=["Model"], ascending=True, inplace=True)

    df.to_csv(os.path.join(PATH_RESULTS, "results_text_" + representation.lower() + ".csv"), index=False)
    df.to_excel(os.path.join(PATH_RESULTS, "results_text_" + representation.lower() + ".xlsx"), index=False)

    output_path = os.path.join(PATH_RESULTS, "results_text_" + representation.lower() + ".@ext")
    plot_acc_f1(df, output_path)


def modeling_w_attributes():
    # load dataset
    logging.info("=" * 50)
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=142,
                                                        train_size=0.8)

    count = 0

    selected_models = {}

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    run_model_selection = True
    if count > 1:
        run_model_selection = False

    base_modeling(X_train, X_test, y_train, y_test, feature_names, dict_results, type_modeling="attr",
                  run_model_selection=run_model_selection, sel_models=selected_models)
    print("")

    data = []

    logging.info("Saving metrics and results")
    for model_name in dict_results.keys():
        accs = dict_results[model_name]["acc"]
        mean_acc = np.median(accs)
        std_dev_acc = np.std(accs)
        f1s = dict_results[model_name]["f1"]
        mean_f1 = np.median(f1s)
        std_dev_f1 = np.std(f1s)

        model = dict_results[model_name]["model"]
        cm = dict_results[model_name]["cm"]
        data.append([model_name, str(model), mean_acc, std_dev_acc, mean_f1, std_dev_f1, cm])

    df = pd.DataFrame(data,
                      columns=["Model", "Model Details", "Mean Acc", "Std Acc", "Mean F1", "Std F1", "Conf Matrix"])
    df.sort_values(by=["Model"], ascending=True, inplace=True)

    df.to_csv(os.path.join(PATH_RESULTS, "results_attr.csv"), index=False)
    df.to_excel(os.path.join(PATH_RESULTS, "results_attr.xlsx"), index=False)

    output_path = os.path.join(PATH_RESULTS, "results_attr.@ext")
    plot_acc_f1(df, output_path)

    # Plot F1


def modeling_w_attributes_and_text(representation):
    # load dataset
    logging.info("=" * 50)
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=142,
                                                        train_size=0.8)

    count = 0
    selected_models = {

    }

    logging.info("=" * 50)

    x_train_bow, bow_model = extract_bow(X_train[:, 1], method=representation)
    x_test_bow = extract_bow(X_test[:, 1], fitted_bow=bow_model, method=representation)
    bow_features = bow_model.get_feature_names_out()

    X_train = np.delete(X_train, 1, 1)  # Remove second column (Conteúdo)
    X_test = np.delete(X_test, 1, 1)  # Remove second column (Conteúdo)

    features_names.extend(bow_features)

    X_train = np.concatenate((X_train, x_train_bow), axis=1)
    X_test = np.concatenate((X_test, x_test_bow), axis=1)

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    run_model_selection = True
    if count > 1:
        run_model_selection = False

    base_modeling(X_train, X_test, y_train, y_test, features_names, dict_results, features_to_select=1000,
                  type_modeling="attr_text_" + representation.lower(), run_model_selection=run_model_selection,
                  sel_models=selected_models)

    print("")

    data = []
    for model_name in dict_results.keys():
        accs = dict_results[model_name]["acc"]
        mean_acc = np.mean(accs)
        std_dev_acc = np.std(accs)
        f1s = dict_results[model_name]["f1"]
        mean_f1 = np.mean(f1s)
        std_dev_f1 = np.std(f1s)
        model = dict_results[model_name]["model"]
        cm = dict_results[model_name]["cm"]
        data.append([model_name, str(model), mean_acc, std_dev_acc, mean_f1, std_dev_f1, cm])

    df = pd.DataFrame(data,
                      columns=["Model", "Model Details", "Mean Acc", "Std Acc", "Mean F1", "Std F1", "Conf Matrix"])
    df.sort_values(by=["Model"], ascending=True, inplace=True)

    df.to_csv(os.path.join(PATH_RESULTS, "results_attr_text_" + representation.lower() + ".csv"), index=False)
    df.to_excel(os.path.join(PATH_RESULTS, "results_attr_text_" + representation.lower() + ".xlsx"), index=False)

    output_path = os.path.join(PATH_RESULTS, "results_attr_text_" + representation.lower() + ".@ext")
    plot_acc_f1(df, output_path)


def build_result_tables():
    df_attr_text = pd.read_csv("modeling/results/results_attr_text_tf-idf.csv")
    df_attr_text_emb = pd.read_csv("modeling/results/results_attr_text_emb.csv")
    df_text = pd.read_csv("modeling/results/results_text_tf-idf.csv")
    df_attr = pd.read_csv("modeling/results/results_attr.csv")
    df_text_emb = pd.read_csv("modeling/results/results_text_emb.csv")

    model_list = sorted(["RF", "MLP", "Naive Bayes", "SVM", "baseline"])
    data_acc = []
    data_f1 = []

    representations = {
        "N-Grams + Attr": df_attr_text,
        "Attr": df_attr,
        "N-Grams": df_text,
        "Embeddings": df_text_emb,
        "Embeddings + Attr": df_attr_text_emb
    }

    for repre in representations.keys():
        line_acc, line_f1 = prepare_row_result(representations[repre], repre, model_list)
        data_acc.append(line_acc)
        data_f1.append(line_f1)

    columns = ["Features"]
    columns.extend(model_list)

    df = pd.DataFrame(data_acc, columns=columns)
    df.to_csv("modeling/results/final_results_acc.csv", index=False)
    df.to_excel("modeling/results/final_results_acc.xlsx", index=False)

    df = pd.DataFrame(data_f1, columns=columns)
    df.to_csv("modeling/results/final_results_f1.csv", index=False)
    df.to_excel("modeling/results/final_results_f1.xlsx", index=False)


def prepare_row_result(df, representation, model_list):
    line_acc = [representation]
    line_f1 = [representation]

    for model_name in model_list:
        row = df.loc[df['Model'] == model_name]

        line_acc.append(round(float(row["Mean Acc"]), 3))
        line_f1.append(round(float(row["Mean F1"]), 3))

    return line_acc, line_f1


def build_hyperparam_tables():
    df_attr_text = pd.read_csv("modeling/results/results_attr_text_tf-idf.csv")
    df_attr_text_emb = pd.read_csv("modeling/results/results_attr_text_emb.csv")
    df_text = pd.read_csv("modeling/results/results_text_tf-idf.csv")
    df_attr = pd.read_csv("modeling/results/results_attr.csv")
    df_text_emb = pd.read_csv("modeling/results/results_text_emb.csv")

    model_list = sorted(["RF", "MLP", "Naive Bayes", "SVM", "baseline"])
    data_acc = []

    representations = {
        "N-Grams + Attr": df_attr_text,
        "Attr": df_attr,
        "N-Grams": df_text,
        "Embeddings": df_text_emb,
        "Embeddings + Attr": df_attr_text_emb
    }

    for repre in representations.keys():
        line = prepare_row_hyperparam(representations[repre], repre, model_list)
        data_acc.append(line)

    columns = ["Features"]
    columns.extend(model_list)

    df = pd.DataFrame(data_acc, columns=columns)
    df.to_csv("modeling/results/final_hyperparam.csv", index=False)
    df.to_excel("modeling/results/final_hyperparam.xlsx", index=False)


def prepare_row_hyperparam(df, representation, model_list):
    line = [representation]

    for model_name in model_list:
        row = df.loc[df['Model'] == model_name]
        line.append(row["Model Details"].item())

    return line


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


def base_modeling(x_train, x_test, y_train, y_test, features_names, dict_results=None, features_to_select=64,
                  type_modeling="", run_model_selection=True, sel_models=dict()):
    if dict_results is None:
        dict_results = dict()

    models = {
        "RF": RandomForestClassifier(n_jobs=8, random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(max_iter=1024, random_state=42),
        "MLP": MLPClassifier(early_stopping=True, shuffle=True, max_iter=512, random_state=42),
    }

    hyper_params = {
        "SVM": {
            "C": [1, 8, 16, 32, 64],
            'gamma': [0.001, 0.01],
            "kernel": ["rbf", "poly"],
            "coef0": [0, 0.01, 0.1, 0.7],
            "decision_function_shape": ["ovo", "ovr"],
            "degree": [1, 3],
            "cache_size": [32, 64, 128]
        },
        "MLP": {
            "hidden_layer_sizes": [
                (32, 32),
                (32, 32, 32),
                (32, 32, 32, 32),
                (64, 64),
                (64, 64, 64),
                (64, 64, 64, 64),
            ],
            "activation": ["relu", "tanh"],
            "batch_size": [32, 64, 128],
            "solver": ["sgd", "adam"],
            "learning_rate": ["constant", "adaptive"],
            "learning_rate_init": [0.01, 0.1, 0.2],
        },
        "Naive Bayes": {
            "var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
        },
        "RF": {
            "n_estimators": [64, 256, 512, 1024],
            "max_depth": [16, 32, 64],
            "max_leaf_nodes": [128, 256, 512, 1024],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2"]
        }
    }

    logging.info("Outliers removal")
    logging.info("Datasamples for each class before remove outliers")
    model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, n_jobs=8)

    print(Counter(y_train))
    predictions = model.fit_predict(x_train, y_train)
    outliers_index = where(predictions != -1)
    x_train = x_train[outliers_index]
    y_train = y_train[outliers_index]

    predict_test = model.predict(x_test)
    outliers_index = where(predict_test != -1)
    x_test = x_test[outliers_index]
    y_test = y_test[outliers_index]

    if x_train.shape[1] > 100:
        logging.info("Running Feature selection")
        if type_modeling.find("emb") >= 0:
            logging.info("Mutual information FS")
            selectFS = SelectKBest(mutual_info_classif, k=100)
        else:
            selectFS = SelectKBest(chi2, k=100)
            logging.info("Chi2 FS")
        x_train = selectFS.fit_transform(x_train, y_train)
        x_test = selectFS.transform(x_test)
        supports = selectFS.get_support(indices=True)
        features = pd.Series(features_names)[supports]
    else:
        features = features_names

    logging.info("Oversampling")
    logging.info("Datasamples for each class before oversampling")
    counter = Counter(y_train)
    logging.info(Counter(y_train))
    # under = RandomUnderSampler(sampling_strategy=0.4)
    # x_train, y_train = under.fit_resample(x_train, y_train)
    over = RandomOverSampler(sampling_strategy=0.4)
    x_train, y_train = over.fit_resample(x_train, y_train)
    logging.info("Datasamples for each class after oversampling/undersampling")
    counter = Counter(y_train)
    logging.info(counter)

    logging.info("Training and testing models")
    count = 0
    for model_name in models.keys():

        logging.info("-" * 50)
        if model_name not in dict_results.keys():
            dict_results[model_name] = {"acc": [], "f1": [], "cm": []}

        model = models[model_name]
        hyper = hyper_params[model_name]

        logging.info("Running model selection for %s" % model_name)
        cv = GridSearchCV(model, hyper, cv=5, verbose=3, n_jobs=8, scoring="f1_macro")
        cv.fit(x_train, y_train)

        models[model_name] = cv.best_estimator_

        logging.info("Best parameters set found on training set:")
        logging.info(cv.best_params_)
        logging.info(cv.best_estimator_)
        logging.info("Results for the best estimator")

        if model_name not in sel_models.keys():
            sel_models[model_name] = cv.best_estimator_
            print(sel_models)

        model = sel_models[model_name]

        logging.info("Testing %s classifier" % model_name)
        logging.info(model)

        y_pred = model.predict(x_test)

        if model_name in ["RF"]:
            features_imp = []
            for feat, imp in zip(features, model.feature_importances_):
                features_imp.append([feat, imp])
            df = pd.DataFrame(features_imp, columns=["Feature", "Importance"])
            df.sort_values(by=["Importance"], ascending=False, inplace=True)
            df.to_excel(os.path.join(PATH_RESULTS, "feature_imp_" + type_modeling + ".xlsx"), index=False)

        acc = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average="macro")

        cm = confusion_matrix(y_test, y_pred)

        dict_results[model_name]["acc"].append(acc)
        dict_results[model_name]["f1"].append(f1)
        dict_results[model_name]["cm"].append(cm)
        dict_results[model_name]["model"] = str(cv.best_params_)

        logging.info("Acc: %.4f \tF1: %.4f" % (acc, f1))

    # Calculate baseline (predict always the majority class)
    len_y_test = len(y_test)
    counter = Counter(y_test)
    most_common_label = counter.most_common(1)[0][0]  # in this case, it is "Preso"

    y_pred = [most_common_label for i in range(len_y_test)]  # Fake a baseline prediction with "Preso"

    acc = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    dict_results["baseline"] = {"acc": [], "f1": [], "cm": []}
    dict_results["baseline"]["acc"].append(acc)
    dict_results["baseline"]["f1"].append(f1)
    dict_results["baseline"]["cm"].append(cm)
    dict_results["baseline"]["model"] = "Majority"
    logging.info("Selected models")
    logging.info(sel_models)

    return dict_results


def run_data_modeling():
    logging.info("=" * 50)
    logging.info("DATA MODELLING")
    plt.rc('axes', axisbelow=True)

    # modeling_w_text_only("TF-IDF")
    # modeling_w_text_only("EMB")
    # modeling_w_attributes_and_text("TF-IDF")
    # modeling_w_attributes_and_text("EMB")
    # modeling_w_attributes()
    # build_result_tables()
    build_hyperparam_tables()


if __name__ == "__main__":
    setup_logging()
    run_data_modeling()
