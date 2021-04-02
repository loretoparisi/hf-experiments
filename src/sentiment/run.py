# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment",
    cache_dir=os.getenv("cache_dir", "../../models"))

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment",
    cache_dir=os.getenv("cache_dir", "../../models"))
