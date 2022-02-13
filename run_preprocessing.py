"""
Run preprocessing script
@date Nov 26, 2021
"""
from preprocessing.merge_datasets import merge_datasets
from preprocessing.preprocess_raw_documents import preprocess_text
from preprocessing.preprocess_spreadsheets import preprocess_spreadsheets_part_i, preprocess_spreadsheets_part_ii
from util.setup_logging import setup_logging


def run_preprocessing():
    # Preprocessing
    # preprocess_text()
    # merge_datasets()
    # preprocess_spreadsheets_part_i()
    preprocess_spreadsheets_part_ii()


if __name__ == '__main__':
    setup_logging()
    run_preprocessing()
