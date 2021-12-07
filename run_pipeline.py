"""
File to run all the pipeline from this project.

@note
    Make sure to place the attributes spreadsheet in the dataset/planilha/
    folder and the folders 'Preso' and 'Solto' in the dataset/raw_docs/ folder.

@date Nov 26,2021
"""
import logging

from eda.eda_part_ii import eda_after_proc_part_i
from eda.eda_part_i import simple_eda
from preprocessing.merge_datasets import merge_datasets
from preprocessing.preprocess_raw_documents import remove_result_from_documents, preprocess_text
from preprocessing.preprocess_spreadsheets import preprocess_spreadsheets_part_i
from util.setup_logging import setup_logging


def run_pipeline():
    # First EDA
    logging.info("-" * 50)
    logging.info("    DATA UNDERSTANDING PART I    ")
    simple_eda()

    logging.info("-" * 50)
    logging.info("    DATA PREPARATION    ")

    # Preprocessing
    preprocess_text()
    merge_datasets()
    preprocess_spreadsheets_part_i()

    logging.info("-" * 50)
    logging.info("    DATA UNDERSTANDING PART II    ")
    # Second EDA
    eda_after_proc_part_i()


if __name__ == '__main__':
    setup_logging()
    run_pipeline()
