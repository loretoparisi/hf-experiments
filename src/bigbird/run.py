# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import torch
from datasets import load_dataset
from transformers import pipeline
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

dataset = load_dataset("patrickvonplaten/scientific_papers_dummy", "arxiv",
    cache_dir=os.getenv("cache_dir", "../../models"))
paper = dataset["validation"]["article"][1]

tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-arxiv",
    cache_dir=os.getenv("cache_dir", "../../models"))
model = AutoModelForSeq2SeqLM.from_pretrained("google/bigbird-pegasus-large-arxiv",
    cache_dir=os.getenv("cache_dir", "../../models"))

summarizer = pipeline(
    'summarization',
    model=model,
    tokenizer=tokenizer)

abstract = summarizer(paper, truncation="longest_first")
print(abstract)