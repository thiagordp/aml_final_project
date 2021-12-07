"""

"""
import logging
import os
import time

import pandas as pd
import seaborn as sns

from eda.eda_part_ii import bag_of_words
from util.constants import PATH_PLANILHA_ATRIBUTOS, PATH_PLANILHA_CRIMES, PATH_PLANILHA_ATTRIB_EXPERT, PATH_RAW_DOCS, PATH_OUTPUT_EDA_I

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="MatplotlibDeprecationWarning")


def simple_eda():
    """

    """

    """
    EDA Attributes
    - Types of data
    - Missing values
    - Central tendency of variables
        - Whole, by label
        - Violin plot
    - Proportion of labels
    - Proportion by relator
    - Proportion by crime
    - Histograms
        - Relator vs labels
    - Processos por tempo (por relator)
    - Correlations
    
    EDA Docs
    - Vocabulary
    - Word cloud
    - Most common words
    - Word cloud by class
    - Word cloud by relator
    """
    logging.info("--------------------------------------------------------")
    logging.info("                     Text analysis")
    logging.info("--------------------------------------------------------")
    # bag_of_words(preprocess_text=False)  # We implemented only one function to prepare the word cloud

    doc_labels = []
    for root, dirs, files in os.walk(PATH_RAW_DOCS, topdown=False):

        label = root.replace(PATH_RAW_DOCS, "").replace(os.sep, "")

        for name in files:
            raw_doc_path = os.path.join(root, name)
            splits_path = raw_doc_path.split(os.sep)

            if len(splits_path) != 4:
                continue
            doc_labels.append(label)

    df = pd.DataFrame(doc_labels, columns=["Resultado"])

    fig, axes = plt.subplots(1, 1, figsize=(8, 4))
    sns.histplot(data=df, x="Resultado", ax=axes)
    # plt.title("HC Results Histogram from a sample")
    plt.tight_layout()
    output_path = os.path.join(PATH_OUTPUT_EDA_I, "histogram_labels.png")
    plt.savefig(output_path)

    logging.info("-" * 60)
    logging.info("                     Spreadsheet analysis")
    logging.info("-" * 60)

    # Loading workbook
    excel_file = pd.ExcelFile(PATH_PLANILHA_ATRIBUTOS)

    logging.info("Available Worksheets for 'Classe-atributos' spreadsheet")
    sheet_names = excel_file.sheet_names
    for name in sheet_names:
        logging.info("\t%s" % name)

    # All the worksheets have metadata from the HC's extracted either by the legal experts or via webscraping.
    # The worksheet 'metadata_import' has metadata from webscraping
    # The others were produced by each of the legal experts in the EGOV research group.

    # In this work, we first describe the worksheet from the first legal expert, "Isabela". The others follow the same structure.
    logging.info("-" * 60)
    logging.info("Analysis for the worksheet '%s'" % sheet_names[0])
    df = excel_file.parse(sheet_names[0])

    logging.info("Basic info: Columns and types")
    logging.info(str(df.info()))

    df = df.sample(frac=1)
    logging.info("Describe the data")
    logging.info(df.describe())

    # Relator
    fig, axes = plt.subplots(1, 1, figsize=(13, 5))
    plt.title("Rapporteur Histogram from a sample")
    df = df.sort_values(by='Relator')
    sns.histplot(data=df, x="Relator", ax=axes, hue="Resultado")
    plt.tight_layout()
    output_path = os.path.join(PATH_OUTPUT_EDA_I, "rapporteur_isabela.png")
    plt.savefig(output_path)
    plt.show()

    logging.info("-" * 50)
    logging.info("Analysis for the worksheet 'metadata_import'")

    df = excel_file.parse("metadata_import")
    df = df.sort_values(by='origem')
    logging.info("Columns and types from metadata_import")
    logging.info(df.info())
    logging.info("First 5 rows")
    logging.info(df.head(n=5))

    fig, axes = plt.subplots(1, 1, figsize=(13, 7))
    plt.title("HC origin from the population")
    sns.histplot(data=df, x="origem", ax=axes)
    plt.xticks(rotation=90)
    axes.set_axisbelow(True)
    plt.grid(axis="y", linestyle=":")
    plt.tight_layout()
    output_path = os.path.join(PATH_OUTPUT_EDA_I, "metadata_info_origin.png")

    plt.savefig(output_path)
    plt.show()

    logging.info("--------------------------------------------------------")
    logging.info("                     Crime spreadsheet analysis")
    logging.info("--------------------------------------------------------")
    excel_file = pd.ExcelFile(PATH_PLANILHA_CRIMES)
    sheet_names = excel_file.sheet_names
    df = excel_file.parse(sheet_names[0])

    print(df.info())
