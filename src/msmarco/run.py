# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os

from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor

class LPModelHelper(object):
    '''
        Helper class for Models implementation
    '''
    
    @classmethod
    def tolabels(self, predictions, int2label_dict, model_type, decimal_places = 4):
        '''
            convert numeric predictions to labeled predictions
        '''
        
        labels = {}
        
        # numpy ndarray
        if model_type=='sklearn' and isinstance(predictions, (np.ndarray, np.generic)):
            for (index, value) in enumerate(predictions):
                label = int2label_dict[index]
                labels[label] = round(value,decimal_places)
        
        # torch tensor
        elif model_type=='bert' and torch.is_tensor(predictions):
            for (index, value) in enumerate(predictions):
                label = int2label_dict[index]
                labels[label] = round(value.item(), decimal_places)
        
        # fasttext (tuple, ndarray)
        elif model_type=='fasttext' and isinstance(predictions, tuple) and isinstance(predictions[1], np.ndarray):
            # predictions are ordered to max prob
            int2label_dict = predictions[0]
            for (index, value) in enumerate(predictions[1]):
                label = int2label_dict[index].replace('__label__', '')
                labels[label] = round(value, decimal_places)
        
        return labels

    @classmethod
    def has_gpu(self):
        '''
            detect gpu available
        '''
        has_gpu = torch.cuda.is_available()
        return has_gpu
    
    @classmethod
    def cosine_similarity(self, a: Tensor, b: Tensor):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        This function can be used as a faster replacement for 1-scipy.spatial.distance.cdist(a,b)
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    #Mean Pooling - Take attention mask into account for correct averaging
    @classmethod
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @classmethod
    def compute_embeddings(self, model,tokenizer, sentences):
        #Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        #Compute query embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        #Perform pooling. In this case, mean pooling
        return self.mean_pooling(model_output, encoded_input['attention_mask'])


# Queries we want embeddings for
queries = ['What is the capital of France?', 'How many people live in New York City?']

# Passages that provide answers
passages = ['Paris is the capital of France', 'New York City is the most populous city in the United States, with an estimated 8,336,817 people living in the city, according to U.S. Census estimates dating July 1, 2019']

#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-MiniLM-L-6-v3",
    cache_dir=os.getenv("cache_dir", "../../models"))
model = AutoModel.from_pretrained("sentence-transformers/msmarco-MiniLM-L-6-v3",
    cache_dir=os.getenv("cache_dir", "../../models"))

# example 1: sentence embedding
query_embeddings = LPModelHelper.compute_embeddings(model, tokenizer, queries)
passage_embeddings = LPModelHelper.compute_embeddings(model, tokenizer, passages)

print("query_embeddings:", query_embeddings.size())
print("passage_embeddings:", passage_embeddings.size())

# cosine similarity
cos_score = LPModelHelper.cosine_similarity(query_embeddings[0], passage_embeddings[0])[0] # list
cos_score = cos_score.cpu() # LP: move to cpu memory
print("cosine_similarity:", cos_score)

# example 2: semantic search
top_k = min(5,len(passages))
similarities = []
query_embedding = LPModelHelper.compute_embeddings(model, tokenizer, queries)
corpus_embeddings = LPModelHelper.compute_embeddings(model, tokenizer, passages)
for embedding in query_embedding:
    cos_scores = LPModelHelper.cosine_similarity(embedding, corpus_embeddings)[0]
    cos_scores = cos_scores.cpu()
    top_results = torch.topk(cos_scores, k=top_k) # find the highest top_k scores
    similarities.append(top_results)

for index,scores in enumerate(similarities):
        print("QUESTION:",queries[index])
        for score, idx in zip(scores[0], scores[1]):
            print("\tPASSAGE:",passages[idx], " (score: %.4f)" % (score))
