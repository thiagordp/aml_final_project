"""
EDA after preprocessing part I

@date December 02, 2021
"""
import os
import time
from collections import Counter

import pandas as pd
import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from preprocessing.preprocess_raw_documents import raw_text_preprocessing, preprocess_text
from util.constants import PATH_PLANILHA_PROC, PATH_OUTPUT_EDA_II, PATH_RAW_DOCS, PATH_OUTPUT_EDA_I, DICT_TRANSLATE_LABEL, DICT_TRANSLATE_CRIME


def most_frequent_crimes():
    df = pd.read_csv(PATH_PLANILHA_PROC.replace("@ext", "csv"))
    df["Enquadramento"] = df["Enquadramento"].replace(["NAN"], "OUTROS")

    # print(df.describe())
    # print(df.info())

    list_crimes = list(df["Enquadramento"])
    list_labels = list(df["Resultado Doc"])
    list_crimes = [[c for c in crime.split(";")] for crime in list_crimes]

    dict_crimes = {}

    for crime_row, label in zip(list_crimes, list_labels):
        value_label = DICT_TRANSLATE_LABEL[label]
        if value_label not in dict_crimes.keys():
            dict_crimes[value_label] = {}

        for crime in crime_row:

            crime_value = DICT_TRANSLATE_CRIME[crime]
            if crime_value in dict_crimes[value_label].keys():
                dict_crimes[value_label][crime_value] += 1
            else:
                dict_crimes[value_label][crime_value] = 1

    print("List crimes", sorted(list(dict_crimes["Released"].keys())))
    data = []
    for value_label in dict_crimes.keys():
        for crime in dict_crimes[value_label].keys():
            data.append([value_label, crime, dict_crimes[value_label][crime]])

    df = pd.DataFrame(data, columns=["Final Outcome", 'Crime', 'Count'])
    df = df.sort_values(by="Crime")

    out_path = os.path.join(PATH_OUTPUT_EDA_II, "crimes_count.xlsx")
    df.to_excel(out_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.grid(ls="--")
    sns.barplot(
        data=df,
        y="Crime", x="Count", hue="Final Outcome",
        palette="deep"
    )
    sns.despine()
    plt.tight_layout()

    out_path = os.path.join(PATH_OUTPUT_EDA_II, "crimes_count_label.png")
    plt.savefig(out_path, dpi=200)

    time.sleep(50)


def most_frequent_crimes_by_year():
    # Maybe place the label here as well
    df = pd.read_csv(PATH_PLANILHA_PROC.replace("@ext", "csv"))
    print("-" * 50)
    print("Most frequent crimes by year")
    print(df.describe())
    print(df.info())

    list_crimes = list(df["Enquadramento"])
    list_years = list(df["ano_documento"])
    # list_labels = list(df["Resultado Doc"])

    list_crimes = [crime.split(";") for crime in list_crimes]

    dict_crimes = {}

    for crime_row, year_doc in zip(list_crimes, list_years):
        if year_doc not in dict_crimes.keys():
            dict_crimes[year_doc] = {}

        for crime in crime_row:
            if crime in dict_crimes[year_doc].keys():
                dict_crimes[year_doc][crime] += 1
            else:
                dict_crimes[year_doc][crime] = 1

    data = []
    for year in dict_crimes.keys():
        for crime in dict_crimes[year].keys():
            data.append([year, crime, dict_crimes[year][crime]])

    df = pd.DataFrame(data, columns=["ano", "crime", "contagem"])
    df = df.sort_values(by="crime")
    out_path = os.path.join(PATH_OUTPUT_EDA_II, "crimes_count_year.xlsx")
    df.to_excel(out_path, index=False)

    fig, ax = plt.subplots(figsize=(17, 8))
    ax.set_xlim(2005, 2021)

    ax.grid(ls="--")
    sns.scatterplot(x="ano",
                    y="crime",
                    size="contagem",
                    sizes=(100, 10000),
                    alpha=0.5,
                    data=df,
                    ax=ax)

    plt.tight_layout()
    plt.legend([], [], frameon=False)

    out_path = os.path.join(PATH_OUTPUT_EDA_II, "crimes_count_year.png")
    plt.savefig(out_path, dpi=200)


def most_frequent_rappourter():
    df = pd.read_csv(PATH_PLANILHA_PROC.replace("@ext", "csv"))
    print("-" * 50)
    print("Most frequent rappourter")
    print(df.describe())
    print(df.info())

    list_rappourter = list(df["Relator"])

    dict_rappourter = {}
    for rapp in sorted(list_rappourter):
        if rapp not in sorted(dict_rappourter.keys()):
            dict_rappourter[rapp] = 1
        else:
            dict_rappourter[rapp] += 1

    data = []
    for key in dict_rappourter.keys():
        data.append([key, dict_rappourter[key]])
    df = pd.DataFrame(data, columns=["Relator", "Contagem"])
    fig, axes = plt.subplots(1, 1, figsize=(14, 9))
    plt.title("Rapporteur Histogram")
    sns.barplot(data=df, x="Contagem", y="Relator", ax=axes, palette=sns.color_palette(['#1f77b4']))
    plt.tight_layout()
    out_path = os.path.join(PATH_OUTPUT_EDA_II, "histogram_rappourter.png")
    plt.savefig(out_path, dpi=200)
    plt.show()


def most_frequent_rappourter_by_label():
    df = pd.read_csv(PATH_PLANILHA_PROC.replace("@ext", "csv"))
    print("-" * 50)
    print("Most frequent rappourter")
    print(df.describe())
    print(df.info())

    dict_rappourter = {}

    for rapp, label in zip(df["Relator"], df["Resultado Doc"]):
        if label not in dict_rappourter.keys():
            dict_rappourter[label] = {}

        if rapp not in dict_rappourter[label].keys():
            dict_rappourter[label][rapp] = 1
        else:
            dict_rappourter[label][rapp] += 1

    data = []
    for label in sorted(dict_rappourter.keys()):
        for rapp in sorted(dict_rappourter[label].keys()):
            data.append([rapp, label, dict_rappourter[label][rapp]])
    df = pd.DataFrame(data, columns=["Relator", "Resultado", "Contagem"])
    fig, axes = plt.subplots(1, 1, figsize=(14, 9))

    plt.title("Rapporteur Histogram by Label")
    sns.barplot(data=df, x="Contagem", y="Relator", hue="Resultado", ax=axes)
    plt.tight_layout()
    out_path = os.path.join(PATH_OUTPUT_EDA_II, "histogram_rappourter_label.png")
    plt.savefig(out_path, dpi=200)
    plt.show()


def most_frequent_subjects():
    df = pd.read_csv(PATH_PLANILHA_PROC.replace("@ext", "csv"))
    print("-" * 50)
    print("Most frequent subjects")

    list_subject = list(df["assuntos"])

    dict_subject = {}
    for rapp in list_subject:
        rapp = rapp.replace("\n", "/").replace("/", "|")
        tokens = [token.strip() for token in rapp.split("|")]
        # print(tokens)

        for token in tokens:

            if token not in sorted(dict_subject.keys()):
                dict_subject[token] = 1
            else:
                dict_subject[token] += 1
    k = Counter(dict_subject)

    # Finding 3 highest values
    high = k.most_common(20)
    data = []

    for key in high:
        data.append([key[0], key[1]])

    df = pd.DataFrame(data, columns=["Assunto", "Contagem"])
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.grid(ls="--")
    plt.title("Rapporteur Histogram")
    sns.barplot(data=df, x="Contagem", y="Assunto", ax=ax, palette=sns.color_palette(['#1f77b4']))

    plt.tight_layout()
    out_path = os.path.join(PATH_OUTPUT_EDA_II, "histogram_assunto.png")
    plt.savefig(out_path, dpi=200)
    plt.show()


def crimes_per_document_per_label():
    df = pd.read_csv(PATH_PLANILHA_PROC.replace("@ext", "csv"))
    print("-" * 50)
    print("Violin plot of crimes per document by label")
    print(df.describe())
    print(df.info())

    dict_rappourter = {}

    for rapp, label in zip(df["Quant"], df["Resultado Doc"]):
        if label not in dict_rappourter.keys():
            dict_rappourter[label] = {}

        if rapp not in dict_rappourter[label].keys():
            dict_rappourter[label][rapp] = 1
        else:
            dict_rappourter[label][rapp] += 1

    data = []
    for label in sorted(dict_rappourter.keys()):
        for rapp in sorted(dict_rappourter[label].keys()):
            data.append([rapp, label, dict_rappourter[label][rapp]])
    df = pd.DataFrame(data, columns=["Quant", "Resultado", "Contagem"])
    fig, axes = plt.subplots(1, 1, figsize=(14, 9))

    plt.title("Quantity Histogram by Label")
    sns.barplot(data=df, y="Contagem", x="Quant", hue="Resultado", ax=axes)
    plt.tight_layout()
    out_path = os.path.join(PATH_OUTPUT_EDA_II, "histogram_crime_quant_label.png")
    plt.savefig(out_path, dpi=200)
    plt.show()


def bag_of_words(preprocess_text=True):
    """We used the same function for before and after processing to avoid duplicating code."""

    dict_text = {}
    string_label_preso = ""
    string_label_solto = ""
    string_full = ""

    for root, dirs, files in os.walk(PATH_RAW_DOCS, topdown=False):

        if root not in dict_text.keys():
            dict_text[root] = ""

        for name in tqdm.tqdm(files):
            raw_doc_path = os.path.join(root, name)
            splits_path = raw_doc_path.split(os.sep)

            if len(splits_path) != 4:
                continue

            with open(raw_doc_path) as fp:

                if preprocess_text:
                    content = []
                    for line in fp:
                        lower_text = line.lower()
                        # TODO: place inside a 'remove line' code
                        if not lower_text.startswith("documento digital") and not lower_text.startswith("documento pode ser acessado no endereço") and \
                                len(lower_text.split()) > 3:
                            content.append(lower_text)

                    text = "\n".join(content)
                    del content
                else:
                    text = fp.read()

            # print(root)
            if root.endswith("Preso"):
                string_label_preso += text + "\n"
            else:
                string_label_solto += text + "\n"
            string_full += text + "\n"

    if preprocess_text:
        string_full = raw_text_preprocessing(string_full)
        string_label_preso = raw_text_preprocessing(string_label_preso)
        string_label_solto = raw_text_preprocessing(string_label_solto)

        outpath_full = os.path.join(PATH_OUTPUT_EDA_II, "wordcloud_full_with_preproc.png")
        outpath_preso = os.path.join(PATH_OUTPUT_EDA_II, "wordcloud_preso_with_preproc.png")
        outpath_solto = os.path.join(PATH_OUTPUT_EDA_II, "wordcloud_solto_with_preproc.png")
    else:
        outpath_full = os.path.join(PATH_OUTPUT_EDA_I, "wordcloud_full_without_preproc.png")
        outpath_preso = os.path.join(PATH_OUTPUT_EDA_I, "wordcloud_preso_without_preproc.png")
        outpath_solto = os.path.join(PATH_OUTPUT_EDA_I, "wordcloud_solto_without_preproc.png")

    _generate_word_cloud(string_full, out_path=outpath_full)
    _generate_word_cloud(string_label_preso, out_path=outpath_preso)
    _generate_word_cloud(string_label_solto, out_path=outpath_solto)


def _generate_word_cloud(text, out_path):
    print("Generating word cloud for ", out_path)
    wordcloud = WordCloud(
        background_color='white', collocations=False, normalize_plurals=False, width=1600, height=900).generate(str(text))
    fig = plt.figure(
        figsize=(16, 9))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(out_path)

    plt.show()
    print("Finished")


def correlation_qty_crimes_result():



    pass


def eda_part_ii():
    most_frequent_crimes()
    most_frequent_crimes_by_year()
    most_frequent_rappourter()
    most_frequent_rappourter_by_label()
    most_frequent_subjects()
    crimes_per_document_per_label()

    bag_of_words(preprocess_text=True)
