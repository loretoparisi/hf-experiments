# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification

model = LukeModel.from_pretrained("studio-ousia/luke-base",
    cache_dir=os.getenv("cache_dir", "../../models"))
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base",
    cache_dir=os.getenv("cache_dir", "../../models"))

text = "Beyoncé lives in Los Angeles."
entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"
inputs = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
outputs = model(**inputs)
word_last_hidden_state = outputs.last_hidden_state
entity_last_hidden_state = outputs.entity_last_hidden_state