"""
File to run all the pipeline from this project.

@note
    Make sure to place the attributes spreadsheet in the dataset/planilha/
    folder and the folders 'Preso' and 'Solto' in the dataset/raw_docs/ folder.

@date Nov 26,2021
"""
import logging

from eda.eda_part_ii import eda_part_ii
from eda.eda_part_i import eda_part_i
from preprocessing.merge_datasets import merge_datasets
from preprocessing.preprocess_raw_documents import remove_result_from_documents, preprocess_text, check_dates
from preprocessing.preprocess_spreadsheets import preprocess_spreadsheets_part_i
from run_data_modeling import modeling_w_text_only, modeling_w_attributes_and_text, modeling_w_attributes, \
    run_data_modeling
from util.setup_logging import setup_logging
import nltk
nltk.download('punkt')


def run_pipeline():
    # First EDA
    logging.info("-" * 50)
    logging.info("    DATA UNDERSTANDING PART I    ")
    eda_part_i()

    logging.info("-" * 50)
    logging.info("    DATA PREPARATION    ")

    # Preprocessing
    preprocess_text()
    merge_datasets()
    preprocess_spreadsheets_part_i()


    # Second EDA
    eda_part_ii()

    run_data_modeling()

if __name__ == '__main__':
    setup_logging()
    run_pipeline()
