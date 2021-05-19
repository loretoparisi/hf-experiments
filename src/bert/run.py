# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
from transformers import BertTokenizer, BertForMaskedLM, BertLMHeadModel, BertForNextSentencePrediction, BertForQuestionAnswering
from transformers import BertModel, BertConfig
from torch.nn import functional as F
import torch

# we will use this tokenizer for all BERT models
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

# encoding = tokenizer.encode_plus(text=question,text_pair=text, add_special_tokens=True)
# inputs = encoding['input_ids']  #Token embeddings
# sentence_embedding = encoding['token_type_ids']  #Segment embeddings
# tokens = tokenizer.convert_ids_to_tokens(inputs) #input tokens
# start_scores, end_scores = model(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
# print(start_scores)
# start_index = torch.argmax(start_scores)
# end_index = torch.argmax(end_scores)
# answer = ' '.join(tokens[start_index:end_index+1])

#inputs = tokenizer.encode_plus(question, text, return_tensors='pt')
# start, end = model(**inputs)
# start_max = torch.argmax(F.softmax(start, dim = -1))
# end_max = torch.argmax(F.softmax(end, dim = -1)) + 1 ## add one ##because of python list indexing
# answer = tokenizer.decode(inputs["input_ids"][0][start_max : end_max])

input_ids = tokenizer.encode(question, text)
print('The input has a total of {:} tokens.'.format(len(input_ids)))
# Search the input_ids for the first instance of the `[SEP]` token.
sep_index = input_ids.index(tokenizer.sep_token_id)
# The number of segment A tokens includes the [SEP] token istelf.
num_seg_a = sep_index + 1
# The remainder are segment B.
num_seg_b = len(input_ids) - num_seg_a
# Construct the list of 0s and 1s.
segment_ids = [0]*num_seg_a + [1]*num_seg_b
# There should be a segment_id for every input token.
assert len(segment_ids) == len(input_ids)
start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                 token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

# Find the tokens with the highest `start` and `end` scores.
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores)

# Combine the tokens in the answer and print it out.
# BERT only needs the token IDs, but for the purpose of inspecting the 
# tokenizer's behavior, let's also get the token strings and display them.
tokens = tokenizer.convert_ids_to_tokens(input_ids)
# Start with the first token.
answer = tokens[answer_start]
# Select the remaining answer tokens and join them with whitespace.
for i in range(answer_start + 1, answer_end + 1):
    # If it's a subword token, then recombine it with the previous token.
    if tokens[i][0:2] == '##':
        answer += tokens[i][2:]
    # Otherwise, add a space then the token.
    else:
        answer += ' ' + tokens[i]
print('Answer: "' + answer + '"')


### Example 6: hidden states embedding
config = BertConfig.from_pretrained('bert-base-uncased',
    cache_dir=os.getenv("cache_dir", "../../models"), output_hidden_states=True)
model = BertModel.from_pretrained('bert-base-uncased',
    cache_dir=os.getenv("cache_dir", "../../models"), config=config)

text = "The capital of France is Paris."
inputs = tokenizer.encode_plus(text, return_tensors = "pt")
outputs = model(**inputs)
'''
 the returns of the BERT model are 
 (last_hidden_state, pooler_output, hidden_states[optional], attentions[optional])
 output[0] is the last hidden state and output[1] is the pooler output.
 output[0] is for a separate representation of each word in the sequence,
 and the pooler is for a joint representation of the entire sequence
'''
print(len(outputs))  # 3

hidden_states = outputs[2]
print(len(hidden_states))  # 13
'''
 Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. 
 The Linear layer weights are trained from the next sentence prediction (classification) objective during pre-training.
 This output is usually not a good summary of the semantic content of the input, 
 you’re often better with averaging or pooling the sequence of hidden-states for the whole input sequence.
'''
embedding_output = hidden_states[0]
attention_hidden_states = hidden_states[1:]
print("shape of embedding layer", embedding_output.size())  # 13