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
model = GENRE.from_pretrained("../models/hf_wikipage_retrieval").eval()
sentences = ["Einstein was a German physicist."]
out=model.sample(
    sentences,
    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
)
print(out)
