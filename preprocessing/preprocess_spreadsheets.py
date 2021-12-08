"""


@date December 2, 2021
"""
import datetime
import logging

import pandas as pd

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
        "Número do HC", "file_name", "num_registro", "pacte_s", "impte_s",
        "proc_a_s_es", "coator_a_s_es", "adv_a_s", "intdo_a_s", "assist_s", "am_curiae",
        "Caminho Doc", "Unnamed: 23", "numero_unico", "relator_ultimo_incidente", "num_na_classe",
        "Conteúdo"  # This one is not required in the spreadsheet, but we will use it later from the txt's.
    ]
    duplicates_columns_to_drop = [
        "Resultado_x", "Crime_x", "Relator_y", "Resultado_y", "Crime_y", "relator"
    ]

    df.drop(useless_columns_to_drop, axis=1, inplace=True)
    df.drop(duplicates_columns_to_drop, axis=1, inplace=True)

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

    df['data_documento'] = pd.to_datetime(df['data_documento'])

    df['ano_documento'] = df['data_documento'].dt.year
    df['assuntos'].fillna("Desconhecido", inplace=True)
    df['Quant'].fillna(1, inplace=True)

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
