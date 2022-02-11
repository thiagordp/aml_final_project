"""

"""
import os

PATH_PLANILHA_ATRIBUTOS = os.path.join("dataset", "planilha", "Classes_Atributos.xlsx")
PATH_PLANILHA_CRIMES = os.path.join("dataset", "planilha", "crimes_enquadrados.xlsx")
PATH_METADATA = os.path.join("dataset", "planilha", "metadata_import.csv")
PATH_PLANILHA_ATTRIB_EXPERT = os.path.join("dataset", "planilha", "Classes_Atributos_Merged.@ext")
PATH_PLANILHA_PROC = os.path.join("dataset", "planilha", "Classes_Atributos_Proc.@ext")
PATH_PLANILHA_RAW_TEXT = os.path.join("dataset", "planilha", "Conteudo_HCs.@ext")
PATH_OUTPUT_EDA_II = os.path.join("results", "eda_ii")
PATH_OUTPUT_EDA_I = os.path.join("results", "eda_i")
RAW_DOCS_FOLDER = "raw_docs"
PROC_DOCS_FOLDER = "proc_docs"
PATH_RAW_DOCS = os.path.join("dataset", RAW_DOCS_FOLDER)
PATH_PROC_DOCS = os.path.join("dataset", PROC_DOCS_FOLDER)

for folder in [PATH_RAW_DOCS, PATH_OUTPUT_EDA_I, PATH_OUTPUT_EDA_II, PATH_PROC_DOCS]:
    if not os.path.exists(folder):
        os.makedirs(folder)

DICT_TRANSLATE_CRIME = {
    'CRIME CONTRA A ADMINISTRAÇÃO PÚBLICA': "CRIME AGAINST THE GOVERNMENT",
    'CRIME CONTRA A DIGNIDADE SEXUAL': "SEXUAL CRIME",
    'CRIME CONTRA A INCOLUMIDADE PAZ E FÉ PÚBLICA': "CRIME AGAINST PUBLIC SAFETY",
    'CRIME CONTRA A PESSOA': "CRIME AGAINST PERSON",
    'CRIME CONTRA O PATRIMÔNIO': "CRIME AGAINST PROPERTY",
    'CRIME DA LEI DE ARMAS': "FIREARMS LAW CRIM",
    'CRIME DA LEI DE DROGAS': "DRUG LAW CRIME",
    'CRIME DE LAVAGEM OU OCULTAÇÃO DE BENS': "LAUNDERING OPERATION CRIME",
    'CRIME DE ORGANIZAÇÃO CRIMINOSA': "CRIMINAL ORGANIZATION",
    'CRIME OUTROS DAS LEIS ESPECIAIS': "OTHER CRIMES",
    'DESCONHECIDO': "UNKNOWN"
}
DICT_TRANSLATE_LABEL = {
    "Preso": "Not released",
    "Solto": "Released"
}
