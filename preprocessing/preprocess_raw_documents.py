"""

"""
import datetime
import json
import logging
import os
import random
import re

import nltk
import pandas as pd
import tqdm
from nltk.corpus import stopwords

nltk.download('stopwords')
from util.constants import PATH_RAW_DOCS, PATH_PLANILHA_RAW_TEXT, RAW_DOCS_FOLDER, PROC_DOCS_FOLDER, PATH_PROC_DOCS


def remove_result_from_documents(source_path, dest_path):
    logging.info("Removing result from documents")
    for root, dirs, files in os.walk(source_path, topdown=False):
        dict_split = {}
        random.shuffle(files)
        for name in files:
            raw_doc_path = os.path.join(root, name)
            splits_path = raw_doc_path.split(os.sep)

            if raw_doc_path.endswith("_proc.txt"):
                os.remove(raw_doc_path)
                continue

            if len(splits_path) == 4:

                content = []
                with open(raw_doc_path, "r", encoding="utf8") as fp:
                    text = fp.read()

                # string_to_split = "     VOTO"
                string_to_split_extrato_ata = "EXTRATO DE ATA"
                text_split = text.split(string_to_split_extrato_ata)
                len_split = len(text_split)

                # Extract judges in the document
                if len_split > 1:
                    text_after_ata = "\n".join(text_split[1:])
                    _extract_judges(text_after_ata)
                    # print(text_after_ata)

                text_before_ata = text_split[0]
                strings_to_split = ["     VOTO", ]
                splits = text_before_ata.split(strings_to_split[0])
                # print(raw_doc_path, len(splits))
                # print(splits[0])

                report_splits = re.split('R[ ]*E[ ]*L[ ]*A[ ]*T[ ]*(O[ ]*|Ó[ ]*)R[ ]*I[ ]*O[ ]*', splits[0])
                # print(report_splits)

                source_docs_folder = source_path.split(os.sep)[-1]
                dest_docs_folder = dest_path.split(os.sep)[-1]

                output_path = raw_doc_path.replace(source_docs_folder, dest_docs_folder)
                folders = os.sep.join(output_path.split(os.sep)[:-1])
                if not os.path.exists(folders):
                    os.makedirs(folders)

                with open(output_path, "w+") as fp:
                    fp.write(report_splits[-1])


def has_ignore_strings(line):
    list_ignore = [
        "documento assinado digitalmente conforme",  # Digital signature message
        "documento pode ser acessado no endereço",  # Digital signature message
        "          supremo tribunal federal",  # Doc header
        "  inteiro teor do acórdão",  # Doc header
        "  voto - "
    ]

    for ignore_str in list_ignore:
        if line.lower().find(ignore_str) >= 0:
            return True

    return False


def remove_useless_headers_and_strings(source_path, dest_path):
    logging.info("Removing useless headers and strings from documents")
    for root, dirs, files in os.walk(source_path, topdown=False):
        dict_split = {}
        random.shuffle(files)
        for name in files:
            raw_doc_path = os.path.join(root, name)
            splits_path = raw_doc_path.split(os.sep)

            if len(splits_path) == 4:
                with open(raw_doc_path, "r", encoding="utf8") as fp:
                    text_lines = fp.readlines()
                new_text = ""

                for line in text_lines:
                    if not has_ignore_strings(line):
                        new_text += line

                dest_file_path = raw_doc_path.replace(source_path, dest_path)
                with open(dest_file_path, "w+", encoding="utf8") as fp:
                    fp.write(new_text)


def _extract_judges(text):
    # TODO implement
    pass


def raw_text_preprocessing(text, remove_stopword=True, stops=None, stemming=False, lemmatization=False):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    token_words = [w for w in tokens if w.isalpha()]
    if remove_stopword:
        if not stops:
            stops = set(stopwords.words('portuguese'))
        token_words = [w for w in token_words if not w in stops]

    joined_words = (" ".join(token_words))

    return joined_words


def raw_corpus_preprocessing(source_path=None, dest_path=None, remove_stopword=True, stemming=False,
                             lemmatization=False):
    logging.info("Text Pre-processing corpus")

    for root, dirs, files in os.walk(source_path, topdown=False):
        dict_split = {}
        random.shuffle(files)
        for name in files:
            raw_doc_path = os.path.join(root, name)
            splits_path = raw_doc_path.split(os.sep)

            if len(splits_path) == 4:
                with open(raw_doc_path, "r", encoding="utf8") as fp:
                    text = fp.read()

                new_text = raw_text_preprocessing(text, remove_stopword=True)

                dest_file_path = raw_doc_path.replace(source_path, dest_path)
                with open(dest_file_path, "w+", encoding="utf8") as fp:
                    fp.write(new_text)


def check_dates():

    dict_file_date = dict()
    for root, dirs, files in os.walk(PATH_RAW_DOCS, topdown=False):
        dict_split = {}
        for name in files:
            raw_doc_path = os.path.join(root, name)
            splits_path = raw_doc_path.split(os.sep)

            if raw_doc_path.endswith("_proc.txt"):
                os.remove(raw_doc_path)
                continue

            if not raw_doc_path.endswith(".txt"):
                continue

            text = open(raw_doc_path, "r").read()

            sub_text = " ".join(text.split()[:50])  # Date should be at the beginning of the file.
            re_result = re.search("([0-9]{2}/[0-9]{2}/[0-9]{4})", sub_text)
            re_result2 = re.search("([0-9]{2}/[0-9]{2}/[0-9]{2})", sub_text)

            if re_result is not None or re_result2 is not None:
                # print("Found", raw_doc_path)
                num_doc = name.replace(".txt", "").upper()

                try:
                    datetime_obj = datetime.datetime.strptime(re_result.group(1), "%d/%m/%Y")
                    dict_file_date[num_doc] = [datetime_obj, re_result.group(1)]
                except:
                    datetime_obj = datetime.datetime.strptime(re_result2.group(1), "%d/%m/%y")
                    if datetime_obj.year > 2020:
                        datetime_obj = datetime_obj.replace(year=datetime_obj.year - 100)
                    dict_file_date[num_doc] = [datetime_obj, re_result2.group(1)]

            else:
                print("Not found:", raw_doc_path)
                dict_file_date[name.replace(".txt", "").upper()] = None

    print(dict_file_date)
    return dict_file_date


def preprocess_text():
    #remove_result_from_documents(PATH_RAW_DOCS, PATH_PROC_DOCS)
    remove_useless_headers_and_strings(PATH_PROC_DOCS, PATH_PROC_DOCS)

    raw_corpus_preprocessing(PATH_PROC_DOCS, PATH_PROC_DOCS)
