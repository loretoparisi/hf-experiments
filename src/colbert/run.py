# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import AutoTokenizer
from transformers import pipeline
from transformers import BertConfig
import torch 
import torch.nn as nn

class VespaColBERT(BertPreTrainedModel):

    def __init__(self,config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, 32, bias=False)
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        Q = self.bert(input_ids,attention_mask=attention_mask)[0]
        Q = self.linear(Q)
        return torch.nn.functional.normalize(Q, p=2, dim=2)  

model = VespaColBERT.from_pretrained("vespa-engine/colbert-medium",
     cache_dir=os.getenv("cache_dir", "../../models"))
tokenizer = AutoTokenizer.from_pretrained(
    "vespa-engine/colbert-medium",
    cache_dir=os.getenv("cache_dir", "../../models"))

# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='tf')
# attention_mask = torch.ones(1,32,dtype=torch.int64)
# input_ids = torch.ones(1,32, dtype=torch.int64)
# args = (input_ids, attention_mask)
# output = model(args)