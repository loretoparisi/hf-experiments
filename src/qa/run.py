# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel

question = 'Marco Aurelio era un imperatore romano che praticava lo stoicismo come filosofia di vita .'
context = 'Manuel Romero è colaborando attivamente con HF / trasformatori per il trader del poder de las últimas técnicas di procesamiento de lenguaje natural al idioma español'

tokenizer = AutoTokenizer.from_pretrained(
    'mrm8488/bert-italian-finedtuned-squadv1-it-alfa',
    cache_dir=os.getenv("cache_dir", "model"))
model = AutoModel.from_pretrained(
    'mrm8488/bert-italian-finedtuned-squadv1-it-alfa',
    cache_dir=os.getenv("cache_dir", "model"))

nlp_qa_bert = pipeline(
    'question-answering',
    model=model,
    tokenizer=tokenizer)

out = nlp_qa_bert({
    'question': question,
    'context': context
})
print(out)

tokenizer = AutoTokenizer.from_pretrained(
    'mrm8488/umberto-wikipedia-uncased-v1-finetuned-squadv1-it',
    cache_dir=os.getenv("cache_dir", "model"))
umberto = AutoModel.from_pretrained(
    'mrm8488/umberto-wikipedia-uncased-v1-finetuned-squadv1-it',
    cache_dir=os.getenv("cache_dir", "model"))

out = nlp_qa_umberto = pipeline(
    'question-answering',
    tokenizer=tokenizer,
    model=umberto)
print(out)



