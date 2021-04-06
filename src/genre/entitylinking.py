# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
from genre.hf_model import GENRE
from genre.trie import Trie
from genre.entity_linking import get_end_to_end_prefix_allowed_tokens_fn_hf as get_prefix_allowed_tokens_fn
from genre.utils import get_entity_spans_hf as get_entity_spans
from genre.utils import get_entity_spans_finalize
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

# entity wikipedia link + .sample to make predictions constraining using prefix_allowed_tokens_fn
def _get_entity_spans(
    model,
    input_sentences,
    prefix_allowed_tokens_fn,
    redirections=None,
):
    output_sentences = model.sample(
        input_sentences,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    )

    output_sentences = [e[0]["text"] for e in output_sentences]
   
    return get_entity_spans_finalize(
        input_sentences, output_sentences, redirections=redirections
    )

sentences = ["Tired of the lies? Tired of the spin? Are you ready to hear the hard-hitting truth in comprehensive, conservative, principled fashion? The Ben Shapiro Show brings you all the news you need to know in the most fast moving daily program in America. Ben brutally breaks down the culture and never gives an inch! Monday thru Friday."]
prefix_allowed_tokens_fn = get_prefix_allowed_tokens_fn(model, sentences)
entity_spans = _get_entity_spans( model, sentences, prefix_allowed_tokens_fn)
print(get_markdown(sentences, entity_spans))