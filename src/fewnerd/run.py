# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import torch

from fewnerd.util.word_encoder import BERTWordEncoder
from fewnerd.model.proto import Proto
from fewnerd.model.nnshot import NNShot

# cache dir
cache_dir = os.getenv("cache_dir", "../../models")
model_name = 'proto'
pretrain_ckpt = 'bert-base-uncased'
max_length = 100

# BERT word encoder
word_encoder = BERTWordEncoder(
        pretrain_ckpt,
        max_length)

if model_name == 'proto':
    # use dot instead of L2 distance for proto
    model = Proto(word_encoder, dot=True)
elif model_name == 'nnshot':
    model = NNShot(word_encoder, dot=False)
elif model_name == 'structshot':
    model = NNShot(word_encoder, dot=False)

load_ckpt = os.path.join(cache_dir, model_name)
state_dict = torch.load(load_ckpt)
own_state = model.state_dict()
for name, param in state_dict.items():
    if name not in own_state:
        print('ignore {}'.format(name))
        continue
    print('load {} from {}'.format(name, load_ckpt))
    own_state[name].copy_(param)