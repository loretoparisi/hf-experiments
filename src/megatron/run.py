# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import torch

from transformers import BertTokenizer, MegatronBertForMaskedLM

# The tokenizer. Megatron was trained with standard tokenizer(s).
tokenizer = BertTokenizer.from_pretrained('nvidia/megatron-bert-uncased-345m',
    cache_dir=os.getenv("cache_dir", "../../models"))
model = MegatronBertForMaskedLM.from_pretrained(
    os.path.join(os.getenv("cache_dir", "../../models"), 'nvidia/megatron-bert-cased-345m'))

# Copy to the device and use FP16.
assert torch.cuda.is_available()
device = torch.device("cuda")
model.to(device)
model.eval()
model.half()

# Create inputs (from the BERT example page).
input = tokenizer("The capital of France is [MASK]", return_tensors="pt").to(device)
label = tokenizer("The capital of France is Paris",  return_tensors="pt")["input_ids"].to(device)

# Run the model.
with torch.no_grad():
    output = model(**input, labels=label)
    print(output)