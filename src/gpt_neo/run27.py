# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

# tokenizer and model with cache_dir
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B",
    cache_dir=os.getenv("cache_dir", "../../models"))
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", 
    cache_dir=os.getenv("cache_dir", "../../models"))

# text generation pipeline
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# generate
out = generator("The U.S. president was ", do_sample=True, min_length=50)
print(out)