"""
File to run all the pipeline from this project.

@note
    Make sure to place the attributes spreadsheet in the dataset/planilha/
    folder and the folders 'Preso' and 'Solto' in the dataset/raw_docs/ folder.

@date Nov 26,2021
"""
from preprocessing.merge_datasets import merge_datasets
from preprocessing.preprocess_spreadsheets import preprocess_spreadsheets_part_i
from util.setup_logging import setup_logging


def run_pipeline():
    # simple_eda()
    merge_datasets()

    preprocess_spreadsheets_part_i()


if __name__ == '__main__':
    setup_logging()
    run_pipeline()
