"""

"""
import logging

import pandas as pd
import seaborn as sns
from util.constants import PATH_PLANILHA_ATRIBUTOS, PATH_PLANILHA_CRIMES, PATH_PLANILHA_ATTRIB_EXPERT

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

    # Loading workbook
    excel_file = pd.ExcelFile(PATH_PLANILHA_ATTRIB_EXPERT.replace("@ext", ".xlsx"))

    logging.info("Available Worksheets")
    sheet_names = excel_file.sheet_names
    logging.info(sheet_names)

    # All the worksheets have metadata from the HC's extracted either by the legal experts or via webscraping.
    # The worksheet 'metadata_import' has metadata from webscraping
    # The others were produced by each of the legal experts in the EGOV research group.

    # In this work, we first describe the worksheet from the first legal expert, "Isabela". The others follow the same structure.
    logging.info("-" * 50)
    logging.info("Analysis for the worksheet '%s'" % sheet_names[0])
    df = excel_file.parse(sheet_names[0])

    logging.info("Columns and types")
    logging.info(df.info())
    logging.info("First 5 rows")
    logging.info(df.head(n=5))

    # Histogram Preso, Solto
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    sns.histplot(data=df, x="Resultado", ax=axes)
    plt.title("HC Results Histogram from a sample")
    plt.tight_layout()
    plt.savefig("eda/result_isabela.png")
    plt.show()
    # Relator
    fig, axes = plt.subplots(1, 1, figsize=(16, 5))
    plt.title("Rapporteur Histogram from a sample")
    sns.histplot(data=df, x="Relator", ax=axes, hue="Resultado")
    plt.tight_layout()
    plt.savefig("eda/rapporteur_isabela.png")
    plt.show()

    logging.info("-" * 50)
    logging.info("Analysis for the worksheet 'metadata_import'")
    # df = excel_file.parse("metadata_import")
    df = df.sort_values(by='origem')
    logging.info("Columns and types")
    logging.info(df.info())
    logging.info("First 5 rows")
    logging.info(df.head(n=5))
    fig, axes = plt.subplots(1, 1, figsize=(16, 7))
    plt.title("HC origin from the population")
    sns.histplot(data=df, x="origem", ax=axes)
    plt.xticks(rotation=90)
    axes.set_axisbelow(True)
    plt.grid(axis="y", linestyle=":")
    plt.tight_layout()
    plt.savefig("eda/origin_population.png")
    plt.show()
    fig, axes = plt.subplots(1, 1, figsize=(16, 5))
    plt.title("Rapporteur Histogram from a sample")
    sns.histplot(data=df, x="relator", ax=axes, hue="Resultado")
    plt.tight_layout()
    plt.savefig("eda/rapporteur_isabela.png")
    plt.show()


    # Central tendency

    # See bag of words

    #
