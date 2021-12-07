"""
File to run all the pipeline from this project.

@note
    Make sure to place the attributes spreadsheet in the dataset/planilha/
    folder and the folders 'Preso' and 'Solto' in the dataset/raw_docs/ folder.

@date Nov 26,2021
"""
from eda.after_proc_eda import eda_after_proc_part_i
from eda.simple_eda import simple_eda
from preprocessing.merge_datasets import merge_datasets
from preprocessing.preprocess_raw_documents import remove_result_from_documents, preprocess_text
from preprocessing.preprocess_spreadsheets import preprocess_spreadsheets_part_i
from util.setup_logging import setup_logging


def run_pipeline():
    # First EDA
    simple_eda()

    # Preprocessing
    #preprocess_text()
    # merge_datasets()
    #preprocess_spreadsheets_part_i()

    # Second EDA
    eda_after_proc_part_i()


if __name__ == '__main__':
    setup_logging()
    run_pipeline()
