# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import pickle
from genre.hf_model import GENRE
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn
from genre.utils import get_entity_spans_hf as get_entity_spans
from genre.utils import get_markdown

cache_dir = os.getenv("cache_dir", "../../models")


# Example: End-to-End Entity Linking
# WIKIPEDIA
# model = GENRE.from_pretrained(os.path.join(cache_dir,"hf_e2e_entity_linking_wiki_abs")).eval()
# WIKIPEDIA + AidaYago2
model = GENRE.from_pretrained(os.path.join(cache_dir,"hf_e2e_entity_linking_aidayago")).eval()

sentences = ["In 1921, Einstein received a Nobel Prize."]

# get the prefix_allowed_tokens_fn with the only constraints to annotate the original sentence (i.e., no other constrains on mention nor candidates)
# use .sample to make predictions constraining using prefix_allowed_tokens_fn
prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(model, sentences)
out = model.sample(
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
# candidate + mention constraint
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

# entity wikipedia link
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
print(get_markdown(sentences, entity_spans)[0])