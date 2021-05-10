# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import torch
from transformers import BigBirdModel, AutoTokenizer

# by default its in `block_sparse` mode with num_random_blocks=3, block_size=64
model = BigBirdModel.from_pretrained("google/bigbird-roberta-large", 
    block_size=64, 
    num_random_blocks=3,
    cache_dir=os.getenv("cache_dir", "../../models"))
tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv",
    cache_dir=os.getenv("cache_dir", "../../models"))

text = "Paris is the <mask> of France."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)