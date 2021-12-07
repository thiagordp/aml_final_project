"""

"""
import json
import logging
import os

import nltk
import tqdm
from nltk.corpus import stopwords

nltk.download('stopwords')
from util.constants import PATH_RAW_DOCS


def remove_result_from_documents():
    for root, dirs, files in os.walk(PATH_RAW_DOCS, topdown=False):
        dict_split = {}
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

                text_before_ata = text_split[0]
                text_without_vote = _remove_vote(text_before_ata)

                strings_to_split = ["     VOTO", ]
                splits = text.split(strings_to_split[0])
                # print(raw_doc_path, len(splits))

                # with open(raw_doc_path.replace(".txt", "_proc.txt"), "w+") as fp:
                #     # fp.write(text_without_vote)
                #     pass

        print(root, dict(sorted(dict_split.items())))


def _remove_vote(text):
    return text


def _extract_judges(text):
    # TODO implement
    pass


def raw_text_preprocessing(text, remove_stopword=True, stemming=False, lemmatization=False):
    logging.info("Starting preprocessing (may take some minutes)")
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    token_words = [w for w in tokens if w.isalpha()]
    if remove_stopword:
        stops = set(stopwords.words('portuguese'))
        token_words = [w for w in token_words if not w in stops]

    joined_words = (" ".join(token_words))

    logging.info("Finished preprocessing")
    return joined_words


def preprocess_text():
    # TODO: Filtering, stemming (?), lemmat(?).. check with related work
    # remove_result_from_documents()
    pass

# TODO: Remove "Documento assinado digitalmente conforme" text
# TODO
