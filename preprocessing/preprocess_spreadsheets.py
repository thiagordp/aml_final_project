"""


@date December 2, 2021
"""
import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from preprocessing.preprocess_raw_documents import check_dates
from util.constants import PATH_PLANILHA_ATTRIB_EXPERT, PATH_PLANILHA_PROC


def remove_unnecessary_features(df: pd.DataFrame):
    """
    The merged dataset has many repeated and/or useless features.
    In this method we remove those features.
    """

    print("-" * 50)
    logging.info("\tPrinting basic info on columns, types and null values:")
    print(df.info())

    logging.info("\tThe dataset has ")

    logging.info("\tAs we can see, there are many columns with NaN values, we deal with them later")
    print("-" * 50)

    # Describe the features in the report
    logging.info("\tDropping columns useless for ML or duplicates")

    # Drop columns
    useless_columns_to_drop = [
        "Número do HC", "file_name", "num_registro", "pacte_s", "impte_s", "tipo_processo", "nivel_sigilo",
        "proc_a_s_es", "coator_a_s_es", "adv_a_s", "intdo_a_s", "assist_s", "am_curiae", "nome_classe",
        "Caminho Doc", "Unnamed: 23", "numero_unico", "relator_ultimo_incidente", "num_na_classe",
    ]
    duplicates_columns_to_drop = [
        "Resultado_x",
        "Resultado_y",
        "Relator_y",
        "Crime_x",
        "Crime_y",
        "relator"
    ]

    df.drop(useless_columns_to_drop, axis=1, inplace=True)
    df.drop(duplicates_columns_to_drop, axis=1, inplace=True)

    df.dropna(subset=["Resultado Doc"], inplace=True)

    return df


def remove_unnecessary_rows(df: pd.DataFrame):
    logging.info("Remove unnecessary rows")

    # Remove docs with result "descartado".
    df = df[df['Resultado Doc'].isin(['Preso', 'Solto'])]
    print(df.info())

    return df


def _preprocess_date(date):
    try:
        timestamp = int(date)
        date = datetime.datetime.fromtimestamp(timestamp)
        return date.strftime("")
    except:
        return date


def rename_columns(df: pd.DataFrame):
    df.rename(columns={'Relator_x': 'Relator', 'old_col2': 'new_col2'}, inplace=True)
    return df


def process_col_values(df: pd.DataFrame):
    df["Relator"] = df["Relator"].apply(lambda x: str(x).upper())
    df["Enquadramento"] = df["Enquadramento"].apply(lambda x: str(x).upper())
    df["Enquadramento"] = df["Enquadramento"].replace(["NAN"], "DESCONHECIDO")

    df['data_documento'] = pd.to_datetime(df['data_documento'], errors="raise")
    df['data_protocolo'] = pd.to_datetime(df['data_protocolo'], format="%d/%m/%Y")

    df['ano_documento'] = df['data_documento'].dt.year
    df['assuntos'].fillna("Desconhecido", inplace=True)
    df['Quant'].fillna(1, inplace=True)
    df["Resultado Doc Num"] = np.where(df['Resultado Doc'] == "Solto", 1, 0)

    nulls = df.isnull().sum()
    print("Null columns\n", nulls[nulls > 0])

    return df


def preprocess_spreadsheets_part_i():
    logging.info("Preprocessing")
    logging.info("Loading file %s" % PATH_PLANILHA_ATTRIB_EXPERT.replace("@ext", "xlsx"))
    df = pd.read_csv(PATH_PLANILHA_ATTRIB_EXPERT.replace("@ext", "csv"))

    df = remove_unnecessary_features(df)
    df = remove_unnecessary_rows(df)
    df = rename_columns(df)
    df = process_col_values(df)

    # df = preprocess_values(df)
    logging.info("Saving the first preprocessing step")
    df.to_csv(PATH_PLANILHA_PROC.replace("@ext", "csv"), index=False)
    df.to_excel(PATH_PLANILHA_PROC.replace("@ext", "xlsx"), index=False)


def one_hot_enconding_for_multiple_value_columns(data_df: pd.DataFrame, col_name: str, sep=";", append_col_name=False):
    """
    Custom One Hot encoding to allow cells with multiple values.
    :param data_df: DataFrame
    :param col_name: Column to onehotenconde
    :param sep: separator char of Cell value
    :return:  None
    """
    col_data = data_df[col_name]

    col_unique_values = list(set(col_data))

    values = []

    # There are rows with more than one value (separated by ";").
    # We need to split those lines to get the actual unique values.
    for col_value in col_unique_values:
        tokens = [token.strip() for token in col_value.replace("\n", sep).split(sep)]
        values.extend(tokens)

    actual_unique_values = list(set(values))
    for new_col in actual_unique_values:
        if append_col_name:
            new_col = col_name + " " + new_col
        data_df[new_col] = 0

    for index, row in data_df.iterrows():
        row_data = str(row[col_name])
        tokens = [token.strip() for token in row_data.replace("\n", sep).split(sep)]

        for token in tokens:
            if append_col_name:
                token = col_name + " " + token
            data_df.at[index, token] = 1

    data_df.drop(columns=[col_name], inplace=True)


def set_extracted_dates(df: pd.DataFrame, col_name: str):
    dict_dates = check_dates()

    for index, row in df.iterrows():
        numero_doc = str(df.iloc[index]["Número do doc"]).upper()
        date_obj = dict_dates[numero_doc]

        df.at[index, col_name] = date_obj[0]

    return df


def preprocess_spreadsheets_part_ii():
    df = pd.read_csv(PATH_PLANILHA_PROC.replace("@ext", "csv"))

    # One Hot Encoding for Crime
    one_hot_enconding_for_multiple_value_columns(df, 'Enquadramento')
    # One Hot enconding for subject
    one_hot_enconding_for_multiple_value_columns(df, 'assuntos', sep="|", append_col_name=True)

    # One Hot Encoding for Rapporteur
    # Aqui é possivel usar o get_dummies porque há apenas um relator por documento
    oh_relator = pd.get_dummies(df.Relator, prefix='Relator')
    df.drop("Relator", axis=1, inplace=True)
    df = df.join(oh_relator)

    oh_cidade = pd.get_dummies(df.origem, prefix="origem")
    df.drop("origem", axis=1, inplace=True)
    df = df.join(oh_cidade)

    df = set_extracted_dates(df, "data_doc_extr")
    df['data_doc_extr'] = pd.to_datetime(df['data_doc_extr'], format="%d/%m/%Y")
    print(df.iloc[2]['data_doc_extr'].strftime("%d/%m/%Y"))
    df['data_protocolo'] = pd.to_datetime(df['data_protocolo'], format="%Y/%m/%d")
    print(df.iloc[2]['data_protocolo'].strftime("%d/%m/%Y"))
    df["diff_datas"] = df["data_doc_extr"] - df["data_protocolo"]
    df['diff_datas'] = pd.to_numeric(df['diff_datas'].dt.days, downcast='integer')

    arr = df["diff_datas"].values
    arr = [val for val in arr if np.isreal(val) and val > 0]

    mean_for_diff_data = np.mean(arr)
    df.loc[(df['diff_datas'] < 0), 'diff_datas'] = mean_for_diff_data
    df['diff_datas'] = df['diff_datas'].fillna(mean_for_diff_data)
    logging.info("Saving the second preprocessing step")
    df.to_csv(PATH_PLANILHA_PROC.replace(".@ext", "_2.csv"), index=False)
    df.to_excel(PATH_PLANILHA_PROC.replace(".@ext", "_2.xlsx"), index=False)
    corr = df.corr()
    corr.to_excel("corr.xlsx")

    # TODO: Missing values imputation
