# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import pickle
from genre.trie import Trie

# load the prefix tree (trie)
with open("../models/kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

from genre.hf_model import GENRE

# Example: Entity Disambiguation
model = GENRE.from_pretrained("../models/hf_entity_disambiguation_aidayago").eval()

sentences = ["Einstein was a [START_ENT] German [END_ENT] physicist."]

# use .sample to make predictions constraining using prefix_allowed_tokens_fn
out = model.sample(
    sentences,
    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
)
print(out)

# Example: Document Retieval
dmodel = GENRE.from_pretrained("../models/hf_wikipage_retrieval").eval()
sentences = ["Einstein was a German physicist."]
out=dmodel.sample(
    sentences,
    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
)
print(out)

# Example: End-to-End Entity Linking
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn
from genre.utils import get_entity_spans_hf as get_entity_spans
lmodel = GENRE.from_pretrained("../models/hf_e2e_entity_linking_aidayago").eval()

# get the prefix_allowed_tokens_fn with the only constraints to annotate the original sentence (i.e., no other constrains on mention nor candidates)
# use .sample to make predictions constraining using prefix_allowed_tokens_fn

sentences = ["In 1921, Einstein received a Nobel Prize."]

prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(model, sentences)

out = lmodel.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
print(out)

# constrain the mentions with a prefix tree (no constrains on candidates)
prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(e)[1:].tolist()
        for e in [" Einstein"]
    ])
)

out = model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
print(out)

# constrain the candidates with a prefix tree (no constrains on mentions)
prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
    model,
    sentences,
    candidates_trie=Trie([
        model.encode(" }} [ {} ]".format(e))[1:].tolist()
        for e in ["Albert Einstein", "Nobel Prize in Physics", "NIL"]
    ])
)
out = model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
print(out)

# constrain the candidate sets given a mention (no constrains on mentions)
prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
    model,
    sentences,
    mention_to_candidates_dict={
        "Einstein": ["Einstein"],
        "Nobel": ["Nobel Prize in Physics"],
    }
)

out = model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
print(out)

# A combiation of these constraints is also possible
prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(" {}".format(e))[1:].tolist()
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)

out = model.sample(
    sentences,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
print(out)

entity_spans = get_entity_spans(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(" {}".format(e))[1:].tolist()
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)
print(entity_spans)

# with the entity_spans generate Markdown with clickable links
from genre.utils import get_markdown

entity_spans = get_entity_spans(
    model,
    sentences,
    mention_trie=Trie([
        model.encode(" {}".format(e))[1:].tolist()
        for e in ["Einstein", "Nobel Prize"]
    ]),
    mention_to_candidates_dict={
        "Einstein": ["Albert Einstein", "Einstein (surname)"],
        "Nobel Prize": ["Nobel Prize in Physics", "Nobel Prize in Medicine"],
    }
)

out = get_markdown(sentences, entity_spans)[0]
print(out)

# Custom End-to-End Entity Linking evaluation
'''
We have some useful function to evaluate End-to-End Entity Linking predictions. Let's suppose we have a Dict[str, str] with document IDs and text as well as the gold entites spans as a List[Tuple[str, int, int, str]] containing documentID, start offset, length and entity title respectively.
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
Then we can get preditions and using get_entity_spans_fairseq to have spans. guess_entities is then a List[List[Tuple[int, int, str]]] containing for each document, a list of entity spans (without the document ID). We further need to add documentIDs to guess_entities and remove the nested list to be compatible with gold_entities
'''
guess_entities = [
    (k,) + x
    for k, e in zip(documents.keys(), guess_entities)
    for x in e
]
# we can import all functions from genre.utils to compute scores.
from genre.utils import (
    get_micro_precision,
    get_micro_recall,
    get_micro_f1,
    get_macro_precision,
    get_macro_recall,
    get_macro_f1,
)

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