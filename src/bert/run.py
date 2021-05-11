# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
from transformers import BertTokenizer, BertForMaskedLM, BertLMHeadModel, BertForNextSentencePrediction, BertForQuestionAnswering
from torch.nn import functional as F
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
    cache_dir=os.getenv("cache_dir", "../../models"))
model = BertForMaskedLM.from_pretrained('bert-base-uncased',
    cache_dir=os.getenv("cache_dir", "../../models"),
    return_dict = True)

### Example 1: Masked Language Modeling
# Masked Language Modeling is the task of decoding a masked token in a sentence.
text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."

input = tokenizer.encode_plus(text, return_tensors = "pt")
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)

output = model(**input)
logits = output.logits

softmax = F.softmax(logits, dim = -1)
mask_word = softmax[0, mask_index, :]

# top ten candidate words
top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
for token in top_10:
   word = tokenizer.decode([token])
   new_sentence = text.replace(tokenizer.mask_token, word)
   print(new_sentence)


# get the top candidate word only
top_word = torch.argmax(mask_word, dim=1)
print(tokenizer.decode(top_word))

### Example 2: Language Modeling
# the task of predicting the best word to follow or continue a sentence given all the words already in the sentence.
model = BertLMHeadModel.from_pretrained('bert-base-uncased',
    return_dict=True, 
    #  is_decoder = True if we want to use this model as a standalone model for predicting the next best word in the sequence. 
    is_decoder = True, 
    cache_dir=os.getenv("cache_dir", "../../models"))

text = "A knife is very "
input = tokenizer.encode_plus(text, return_tensors = "pt")
output = model(**input).logits[:, -1, :]
softmax = F.softmax(output, -1)
index = torch.argmax(softmax, dim = -1)
x = tokenizer.decode(index)
print(text + " " + x)

### Example 4: Next Sentence Prediction
# Next Sentence Prediction is the task of predicting whether one sentence follows another sentence. 
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased',
    cache_dir=os.getenv("cache_dir", "../../models"))
prompt = "The child came home from school."
next_sentence = "He played soccer after school."
encoding = tokenizer.encode_plus(prompt, next_sentence, return_tensors='pt')
outputs = model(**encoding)[0]
softmax = F.softmax(outputs, dim = 1)
print(softmax)

### Example 5: Extractive Question Answering
# Extractive Question Answering is the task of answering a question given some context text by outputting the start and end indexes of where the answer lies in the context.

model = BertForQuestionAnswering.from_pretrained('bert-base-uncased',
    cache_dir=os.getenv("cache_dir", "../../models"))
question = "What is the capital of France?"
text = "The capital of France is Paris."
inputs = tokenizer.encode_plus(question, text, return_tensors='pt')
start, end = model(**inputs)
start_max = torch.argmax(F.softmax(start, dim = -1))
end_max = torch.argmax(F.softmax(end, dim = -1)) + 1 ## add one ##because of python list indexing
answer = tokenizer.decode(inputs["input_ids"][0][start_max : end_max])
print(answer)