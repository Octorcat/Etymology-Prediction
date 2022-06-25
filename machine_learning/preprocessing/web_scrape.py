# ----------------------------------------------------------------------------
# Created By  : Frederick Roman
# License : MIT
# ---------------------------------------------------------------------------
""" 
Run 'python preprocessing/web_scrape.py' to scrape Wiktionary 
for etymologies and store the results to collected_etymology_dict.json
"""
from nltk.stem import WordNetLemmatizer
from wiktionaryparser import WiktionaryParser
from nltk import regexp_tokenize
import json
import os

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
ETYMOLOGY = Literal["Latin", "Germanic borrowing of Latin", "Proto-Germanic", "unknown"]


CWD = os.getcwd()
DATA_FILE_DIR = f"{CWD}/machine_learning/data/CMU_dict.txt"
CMU_SRC_DICT_PATH = f"{CWD}/machine_learning/preprocessing/collected_etymology_dict.json"
ETYMOLOGY_REGEX = r"""Proto-Germanic|Germanic borrowing of Latin|Latin"""

Lemmatizer = WordNetLemmatizer()
Parser = WiktionaryParser()
Parser.set_default_language("english")


def get_word_source(query_word: str) -> ETYMOLOGY:
    """
    Get word source etymology by parsing Wiktionary's etymology entry.
    Since Wiktionary's etymology entry is written as an unstructured
    human-readable text, the etymology class must extracted through a regex.

    :return: word source etymology
    """
    try:
        # same lexeme => same source
        lemmatized_word = Lemmatizer.lemmatize(query_word)
        # fetch Wiktionary and take the first result
        word_entry = Parser.fetch(lemmatized_word)[0]
        # take the first etymology mentioned (which you assume to be the source)
        word_source = regexp_tokenize(word_entry["etymology"], ETYMOLOGY_REGEX)[0]
        return word_source or "unknown"
    except:
        return "unknown"


def write_word_source_dict() -> None:
    """
    Read CMU_dict of English words.
    Fetch source etymology of each of its words.
    Write the results to collected_etymology_dict.
    """
    word_source_dict = {}
    with open(DATA_FILE_DIR) as input_file:
        for line in input_file:
            try:
                word = (
                    line.strip()
                    .lower()  # disregard casing for etymology
                    .split()[0]  # take the word-writting section
                    # remove word-variants substrings
                    .replace("(2)", "")
                    .replace("(3)", "")
                    .replace("(4)", "")
                )
                source = get_word_source(word)
                print(f"{word}: {source}")
                word_source_dict[word] = source
            except:
                continue
    with open(CMU_SRC_DICT_PATH, "w") as output_file:
        json.dump(word_source_dict, output_file)


write_word_source_dict()
