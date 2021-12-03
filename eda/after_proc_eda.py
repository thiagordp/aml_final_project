"""
EDA after preprocessing part I

@date December 02, 2021
"""
import os

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from util.constants import PATH_PLANILHA_PROC, PATH_OUTPUT_EDA


def most_frequent_crimes():
    df = pd.read_csv(PATH_PLANILHA_PROC.replace("@ext", "csv"))
    df["Enquadramento"] = df["Enquadramento"].replace(["NAN"], "OUTROS")

    print(df.describe())
    print(df.info())

    list_crimes = list(df["Enquadramento"])
    list_labels = list(df["Resultado Doc"])
    list_crimes = [[c for c in crime.split(";")] for crime in list_crimes]

    dict_crimes = {}

    for crime_row, label in zip(list_crimes, list_labels):
        if label not in dict_crimes.keys():
            dict_crimes[label] = {}

        for crime in crime_row:
            if crime in dict_crimes[label].keys():
                dict_crimes[label][crime] += 1
            else:
                dict_crimes[label][crime] = 1
    data = []
    for label in dict_crimes.keys():
        for crime in dict_crimes[label].keys():
            data.append([label, crime, dict_crimes[label][crime]])

    df = pd.DataFrame(data, columns=["Resultado", 'Crime', 'Contagem'])
    out_path = os.path.join(PATH_OUTPUT_EDA, "crimes_count.xlsx")
    df.to_excel(out_path, index=False)

    fig, ax = plt.subplots(figsize=(15, 7))

    ax.grid(ls="--")
    sns.barplot(
        data=df,
        y="Crime", x="Contagem", hue="Resultado"
    )
    sns.despine()
    plt.tight_layout()

    out_path = os.path.join(PATH_OUTPUT_EDA, "crimes_count_label.png")
    plt.savefig(out_path, dpi=200)


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

    print(list_crimes)
    print(dict_crimes)

    data = []
    for year in dict_crimes.keys():
        for crime in dict_crimes[year].keys():
            data.append([year, crime, dict_crimes[year][crime]])

    df = pd.DataFrame(data, columns=["ano", "crime", "contagem"])
    out_path = os.path.join(PATH_OUTPUT_EDA, "crimes_count_year.xlsx")
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

    out_path = os.path.join(PATH_OUTPUT_EDA, "crimes_count_year.png")
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
    out_path = os.path.join(PATH_OUTPUT_EDA, "histogram_rappourter.png")
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
    df = pd.DataFrame(data, columns=["Relator","Resultado", "Contagem"])
    fig, axes = plt.subplots(1, 1, figsize=(14, 9))

    plt.title("Rapporteur Histogram by Label")
    sns.barplot(data=df, x="Contagem", y="Relator",hue="Resultado", ax=axes)
    plt.tight_layout()
    out_path = os.path.join(PATH_OUTPUT_EDA, "histogram_rappourter_label.png")
    plt.savefig(out_path, dpi=200)
    plt.show()


def eda_after_proc_part_i():
    most_frequent_crimes()
    most_frequent_crimes_by_year()
    most_frequent_rappourter()
    most_frequent_rappourter_by_label()