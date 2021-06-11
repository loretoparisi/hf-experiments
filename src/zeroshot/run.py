# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os,sys
import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from numpy import argmax

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(BASE_PATH, '..'))
from lpdutils.lpdutils import LPDUtils

# tokenizer and model with cache_dir
tokenizer = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli",
    cache_dir=os.getenv("cache_dir", "../../models"))
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli", 
    cache_dir=os.getenv("cache_dir", "../../models"))

if LPDUtils.has_gpu():
    classifier = pipeline("zero-shot-classification", 
        model=model,
        tokenizer=tokenizer,
        device=0) # to utilize GPU
else:
    classifier = pipeline("zero-shot-classification", 
        model=model,
        tokenizer=tokenizer)

sequence = "one day I will see the world"

# example 1: single-class
candidate_labels = ['travel', 'cooking', 'dancing']
res = classifier(sequence, candidate_labels)

scores = res["scores"]
classes = res["labels"]
best_index = argmax(scores)
predicted_class = classes[best_index]
predicted_score = scores[best_index]
print(predicted_class, predicted_score)

# example 2: multi-class
sequence = "I like swimming at the seaside"
candidate_labels = ['sports', 'traveling', 'summer', 'winter', 'politics']
res = classifier(sequence, candidate_labels, multi_label=True)
for index,label in enumerate(res["labels"]):
    print(label, res["scores"][index])

# example 3: hypothesis translation
# default hypothesis template is ""This text is {}"". 
# within one language, translate the hypothesis to that language
sequence = "¿A quién vas a votar en 2020?"
candidate_labels = ["Europa", "salud pública", "política"]
hypothesis_template = "Este ejemplo es {}."
res = classifier(sequence, candidate_labels, hypothesis_template=hypothesis_template)
print(res)

# example 4: manual entailment
device = LPDUtils.get_device()
premise = sequence
hypothesis = "This example is {}."

# run through model pre-trained on MNLI
x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
                     truncation_strategy='only_first')
logits = model(x.to(device))[0]

# we throw away "neutral" (dim 1) and take the probability of
# "entailment" (2) as the probability of the label being true 
entail_contradiction_logits = logits[:,[0,2]]
probs = entail_contradiction_logits.softmax(dim=1)
prob_true_label = probs[:,1] # taking all rows (:) but keeping the second column (1)
with torch.no_grad():
    best_index = argmax(probs,axis=1)
    probs = [t.numpy() for t in probs][0]
    prob_true_label = probs[best_index]
    print(prob_true_label)