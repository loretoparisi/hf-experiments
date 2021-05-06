# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification, LukeForEntitySpanClassification

###
# LUKE models in the following examples will be saved to cache_dir=../../models
#
# studio-ousia/luke-base
# studio-ousia/luke-large-finetuned-tacred
# studio-ousia/luke-large-finetuned-conll-2003
###

model = LukeModel.from_pretrained("studio-ousia/luke-base",
    cache_dir=os.getenv("cache_dir", "../../models"))
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base",
    cache_dir=os.getenv("cache_dir", "../../models"))

# Example 1: Computing the contextualized entity representation corresponding to the entity mention "Beyoncé"
text = "Beyoncé lives in Los Angeles."
entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"
inputs = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
outputs = model(**inputs)
word_last_hidden_state = outputs.last_hidden_state
entity_last_hidden_state = outputs.entity_last_hidden_state

# Example 2: Inputting Wikipedia entities to obtain enriched contextualized representations
entities = ["Beyoncé", "Los Angeles"]  # Wikipedia entity titles corresponding to the entity mentions "Beyoncé" and "Los Angeles"
entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
inputs = tokenizer(text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
outputs = model(**inputs)
word_last_hidden_state = outputs.last_hidden_state
entity_last_hidden_state = outputs.entity_last_hidden_state

# word_last_hidden_state: torch.Size([1, 9, 768])  entity_last_hidden_state: torch.Size([1, 2, 768])
print("word_last_hidden_state:", word_last_hidden_state.size(), " entity_last_hidden_state:", entity_last_hidden_state.size())

# Example 3: Classifying the relationship between two entities using LukeForEntityPairClassification head model
large_model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred",
    cache_dir=os.getenv("cache_dir", "../../models"))
large_tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred",
    cache_dir=os.getenv("cache_dir", "../../models"))

entity_spans = [(0, 7), (17, 28)]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
inputs = large_tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
outputs = large_model(**inputs)
logits = outputs.logits
predicted_class_idx = int(logits[0].argmax())
# Predicted class: per:cities_of_residence
print("Predicted class:", large_model.config.id2label[predicted_class_idx])

# Example 4: The LUKE model with a span classification head on top (a linear layer on top of the hidden states output) for tasks such as named entity recognition.
ner_model = LukeForEntitySpanClassification.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003",
    cache_dir=os.getenv("cache_dir", "../../models"))
ner_tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-conll-2003",
    cache_dir=os.getenv("cache_dir", "../../models"))

# List all possible entity spans in the text
word_start_positions = [0, 8, 14, 17, 21]  # character-based start positions of word tokens
word_end_positions = [7, 13, 16, 20, 28]  # character-based end positions of word tokens
entity_spans = []
for i, start_pos in enumerate(word_start_positions):
     for end_pos in word_end_positions[i:]:
         entity_spans.append((start_pos, end_pos))

inputs = ner_tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
outputs = ner_model(**inputs)
predicted_class_idx = int(outputs.logits[0].argmax())
print("Predicted class:", ner_model.config.id2label[predicted_class_idx])

# Example 5: LUKE large finetuned 
inputs = [
    { "text": "Lysandre lives in New York City", "spans": [(0,7),(18,31)]},
    { "text": "Hugging Face's model hub is available at huggingface.co", "spans": [(0,12),(41,51)]},
    { "text": "Hugging Face is also called HuggingFace or HF, according to who you ask", "spans": [(0,12),(43,45)]}
]
for input_data in inputs:
    inputs = large_tokenizer(input_data['text'], entity_spans=input_data['spans'], return_tensors="pt")
    outputs = large_model(**inputs)
    predicted_class_idx = int(outputs.logits[0].argmax())
    print(f"[Predicted class: {large_model.config.id2label[predicted_class_idx]}] {input_data['text']}")