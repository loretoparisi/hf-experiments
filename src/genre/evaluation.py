# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

from genre.hf_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn
from genre.utils import get_entity_spans_hf as get_entity_spans
# import all functions from genre.utils to compute scores.
from genre.utils import (
    get_micro_precision,
    get_micro_recall,
    get_micro_f1,
    get_macro_precision,
    get_macro_recall,
    get_macro_f1,
)

model = GENRE.from_pretrained("../models/hf_e2e_entity_linking_aidayago").eval()

# Example: Custom End-to-End Entity Linking evaluation
'''
We have some useful function to evaluate End-to-End Entity Linking predictions. 
Let's suppose we have a Dict[str, str] with document IDs and text as well as the gold entites spans as a List[Tuple[str, int, int, str]] containing documentID, start offset, length and entity title respectively.
'''
documents = {
    "id_0": "In 1921, Einstein received a Nobel Prize.",
    "id_1": "Armstrong was the first man on the Moon.",
}

gold_entities = [
    ("id_0", 3, 4, "1921"),
    ("id_0", 9, 8, 'Albert_Einstein'),
    ("id_0", 29, 11, 'Nobel_Prize_in_Physics'),
    ("id_1", 0, 9, 'Neil_Armstrong'),
    ("id_1", 35, 4, 'Moon'),
]
guess_entities = get_entity_spans(
    model,
    list(documents.values()),
)
'''
Then we can get preditions and using get_entity_spans to have spans. guess_entities is then a List[List[Tuple[int, int, str]]] 
containing for each document, a list of entity spans (without the document ID). We further need to add documentIDs to guess_entities and remove the nested list to be compatible with gold_entities
'''
guess_entities = [
    (k,) + x
    for k, e in zip(documents.keys(), guess_entities)
    for x in e
]

# Finally, we use genre.utils functions to compute scores.
micro_p = get_micro_precision(guess_entities, gold_entities)
micro_r = get_micro_recall(guess_entities, gold_entities)
micro_f1 = get_micro_f1(guess_entities, gold_entities)
macro_p = get_macro_precision(guess_entities, gold_entities)
macro_r = get_macro_recall(guess_entities, gold_entities)
macro_f1 = get_macro_f1(guess_entities, gold_entities)

print(
   "micro_p={:.4f} micro_r={:.4f}, micro_f1={:.4f}, macro_p={:.4f}, macro_r={:.4f}, macro_f1={:.4f}".format(
       micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1
   )
)