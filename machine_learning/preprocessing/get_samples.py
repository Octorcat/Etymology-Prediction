# ----------------------------------------------------------------------------
# Created By  : Frederick Roman
# License : MIT
# ---------------------------------------------------------------------------
""" 
Run 'python preprocessing/get_samples.py' to read CMU_source_dict.json,
group the sample words by etymology, and 
balance the lists of latin and germanic ones for training.
"""
import os
import json
import random
from typing import Dict, List, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
ETYMOLOGY = Literal["Latin", "Proto-Germanic", "unknown"]
Words = List[str]

CMU_SRC_DICT_PATH = f"{os.getcwd()}/preprocessing/CMU_source_dict.json"


def group_words_by_etymology() -> Dict[ETYMOLOGY, Words]:
    """
    Group words in CMU source dictionary by etymology.

    :return: etymological_groups
    """
    with open(CMU_SRC_DICT_PATH) as file:
        etymology_dict = json.load(file)
        etymological_groups = {"Latin": [], "Proto-Germanic": [], "unknown": []}
        for word, etymology in etymology_dict.items():
            if etymology == "Latin":
                etymological_groups["Latin"].append(word)
            elif etymology == "Proto-Germanic":
                etymological_groups["Proto-Germanic"].append(word)
            elif etymology == "unknown":
                etymological_groups["unknown"].append(word)
        return etymological_groups


def balance_lists(list_a: Words, list_b: Words) -> Tuple[Words, Words]:
    """
    Balance list_a and list_b to have the same length.
    Sample the longer list randomly to match the shorter one's length.

    :return: list_a, list_b where len(list_a) == len(list_b)
    """
    nb_list_a = len(list_a)
    nb_list_b = len(list_b)
    if nb_list_a > nb_list_b:
        quot = nb_list_a // (nb_list_b or 1)
        rem = nb_list_a % nb_list_b
        list_b = list(list_b * quot + random.sample(list_b, rem))
    else:
        quot = nb_list_b // (nb_list_a or 1)
        rem = nb_list_b % nb_list_a
        list_a = list(list_a * quot + random.sample(list_a, rem))
    return list_a, list_b


def balance_samples_for_training(groups: Dict[ETYMOLOGY, Words]) -> Tuple[Words, Words]:
    """
    Filter out 'unknown's (by not selecting them) and
    balance samples for etymology classification training.

    :return: latin_list, germanic_list where len(latin_list) == len(germanic_list)
    """
    latin_list = groups["Latin"]
    germanic_list = groups["Proto-Germanic"]
    return balance_lists(latin_list, germanic_list)


etymological_groups: Dict[ETYMOLOGY, Words] = group_words_by_etymology()
latin_list, germanic_list = balance_samples_for_training(etymological_groups)
