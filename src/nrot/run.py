# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration

context = """Saint Jean de Brébeuf was a French Jesuit missionary who
travelled to New France in 1625. There he worked primarily with the Huron
for the rest of his life, except for a few years in France from 1629 to
1633. He learned their language and culture, writing extensively about
each to aid other missionaries. In 1649, Br´ebeuf and another missionary
were captured when an Iroquois raid took over a Huron village . Together
with Huron captives, the missionaries were ritually tortured and killed
on March 16, 1649. Brébeuf was beatified in 1925 and among eight Jesuit
missionaries canonized as saints in the Roman Catholic Church in 1930."""

questions = [
    "When Brébeuf was beatified?", # T5 Answer:  <pad> 1 9 2 5 </s>
    "When Brébeuf was canonized?", # T5 Answer:  <pad> 1 9 3 0 </s>
    "With how many missionaries was canonized?", # T5 Answer:  <pad> 8 </s>
    "With how many missionaries was Brébeuf canonized?", # T5 Answer:  <pad> 1 7 </s>
    "How many missionaries were canonized?", # T5 Answer:  <pad> 8 </s>
    "How many missionaries were canonized as saints?", # T5 Answer:  <pad> 1 7 </s>
]

tokenizer = T5Tokenizer.from_pretrained("nielsr/nt5-small-rc1",
    cache_dir=os.getenv("cache_dir", "../../models"))
model = T5ForConditionalGeneration.from_pretrained("nielsr/nt5-small-rc1",
    cache_dir=os.getenv("cache_dir", "../../models"))

for question in questions:
    # encode context & question
    input_text = f"answer_me: {question} context: {context}"
    encoded_query = tokenizer(
                        input_text, 
                        return_tensors='pt', 
                        padding='max_length', 
                        truncation=True, 
                        max_length=512)

    # generate answer
    generated_answer = model.generate(input_ids=encoded_query["input_ids"], 
                                    attention_mask=encoded_query["attention_mask"], 
                                    max_length=54)

    decoded_answer = tokenizer.decode(generated_answer.numpy()[0])
    print("T5 Answer: ", decoded_answer)