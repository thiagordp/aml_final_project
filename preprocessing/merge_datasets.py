"""
Code to merge spreadsheets and text files into one CSV

@author Thiago Dal Pont
@date November 29, 2021
"""
import logging
import os
import time

import pandas as pd

from util.constants import PATH_PLANILHA_ATRIBUTOS, PATH_PLANILHA_CRIMES, PATH_PLANILHA_ATTRIB_EXPERT, PATH_RAW_DOCS, \
    PATH_PLANILHA_RAW_TEXT, PATH_METADATA, PATH_PLANILHA_PROC, PROC_DOCS_FOLDER, PATH_PROC_DOCS


def unify_attrib_worksheets():
    print("Merge legal experts' attribute worksheets")
    # Load spreadsheet
    excel_file_attrib = pd.ExcelFile(PATH_PLANILHA_ATRIBUTOS)

    # Get worksheet names
    sheet_names = list(excel_file_attrib.sheet_names)

    # Remove attributes automatically extracted via scraping
    # and keep those extracted by the legal experts
    sheet_names.remove("metadata_import")

    cols = []
    list_df = []  # list of df's

    # Iterate over list of worksheet names
    for sheet_name in sheet_names:
        df = excel_file_attrib.parse(sheet_name)
        cols.extend(list(df.columns))
        list_df.append(df)

    # Concatenate the df's rows.
    merged_df = pd.concat(list_df, ignore_index=True)

    # Select columns
    # (In the labelling and attributes extraction steps, some legal experts created additional columns which will not be used in this project)
    merged_df = merged_df[["Número do doc", "Número do HC", "Resultado", "Crime", "Relator"]]
    merged_df["Número do doc"] = merged_df["Número do doc"].apply(lambda x: x.upper())

    # Get df from worksheet of attrib extracted via scraping
    autoextract_attrib = pd.read_csv(PATH_METADATA, sep=";")
    autoextract_attrib["Número do doc"] = autoextract_attrib["Número do doc"].apply(lambda x: x.upper())
    autoextract_attrib["data_documento"] = autoextract_attrib["data_documento"].apply(lambda x: str(x))

    # Merge columns based on "Número do doc" column
    merged_auto_df = merged_df.merge(autoextract_attrib, on="Número do doc")

    print("-" * 50)
    excel_file_crimes = pd.ExcelFile(PATH_PLANILHA_CRIMES)
    crimes_df = excel_file_crimes.parse("Inferidos")
    crimes_df["Número do doc"] = crimes_df["Número do doc"].apply(lambda x: x.upper())

    relator_list = sorted(set(crimes_df["Relator"]))

    df_sample = crimes_df.sample(n=10)

    merged_crime_df = merged_auto_df.merge(crimes_df, how="left", on="Número do doc")

    excel_file_content = pd.ExcelFile(PATH_PLANILHA_RAW_TEXT.replace("@ext", "xlsx"))
    content_df = excel_file_content.parse("Sheet1")
    final_df = merged_crime_df.merge(content_df, how="left", on="Número do doc")

    # Merge text from docs
    final_df.to_csv(PATH_PLANILHA_ATTRIB_EXPERT.replace("@ext", "csv"), index=False)
    final_df.to_excel(PATH_PLANILHA_ATTRIB_EXPERT.replace("@ext", "xlsx"), index=False)

    return merged_df


def merge_text_files_into_spreadsheet(source_path):
    print("Merge raw docs into spreadsheet")
    final_data = []

    # Walk through the dirs and list the raw docs.
    aux_count = 0
    for root, dirs, files in os.walk(source_path, topdown=False):

        for name in files:
            raw_doc_path = os.path.join(root, name)
            splits_path = raw_doc_path.split(os.sep)

            if len(splits_path) == 4:
                with open(raw_doc_path, "r", encoding="utf8") as fp:
                    text = fp.read()

                label = splits_path[2]
                file_name = splits_path[3].replace(".txt", "").upper()
                data = [file_name, label, raw_doc_path, text]
                final_data.append(data)
                if aux_count % 500 == 0:
                    print("=", end="")

            aux_count += 1

    df = pd.DataFrame(final_data, columns=[
        "Número do doc",
        "Resultado Doc",
        "Caminho Doc",
        "Conteúdo"
    ])

    df.to_csv(PATH_PLANILHA_RAW_TEXT.replace("@ext", "csv"), index=False)
    df.to_excel(PATH_PLANILHA_RAW_TEXT.replace("@ext", "xlsx"), index=False)


def merge_datasets():
    logging.info("Merging datasets")
    merge_text_files_into_spreadsheet(PATH_PROC_DOCS)
    unify_attrib_worksheets()
