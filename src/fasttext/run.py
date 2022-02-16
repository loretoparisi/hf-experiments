# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2022 Loreto Parisi (loretoparisi at gmail dot com)

import os
from huggingface_hub import hf_hub_download
import fasttext

def prob2labels(predictions, decimal_places=2):
    labels = {}
    # predictions are ordered to max prob
    int2label_dict = predictions[0]
    for (index, value) in enumerate(predictions[1]):
        label = int2label_dict[index].replace('__label__', '')
        labels[label] = round(value, decimal_places)
    return labels

model_path = hf_hub_download("julien-c/fasttext-language-id", "lid.176.bin", 
        cache_dir=os.getenv("cache_dir", "../../models"))

# Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
# LP: ignore this Warning: https://fasttext.cc/blog/2019/06/25/blog-post.html
model = fasttext.load_model(model_path)

sequence = "The head of the United Nations says there is no military solution in Syria"
num_labels = 3
predictions = model.predict(sequence, k=num_labels)

print(sequence, prob2labels(predictions))