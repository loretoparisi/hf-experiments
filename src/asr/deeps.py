# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)
# Code adpated from https://pastebin.com/3wWj59uz
# Code adpated from [Nikita Schneider](https://twitter.com/DeepSchneider) https://twitter.com/DeepSchneider/status/1381179738824314880?s=20

import os
import librosa
import torch
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import deepspeed

#tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h", cache_dir=os.getenv("cache_dir", "../models"))
#model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h", cache_dir=os.getenv("cache_dir", "../models"))

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", 
    cache_dir=os.getenv("cache_dir", "../models"))
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", 
    cache_dir=os.getenv("cache_dir", "../models"))

# DeepSpeed config
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"
config = {
    "train_batch_size": 8,
    "fp16": {
        "enabled": True,
        "min_loss_scale": 1,
        "opt_level": "O3"
    },
    "zero_optimization": {
        "stage": 2,
        "cpu_offload": True,
        "cpu_offload_params": True,
        "contiguous_gradients": True,
        "overlap_comm": True
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001,
            "betas": [0.9, 0.999],
            "eps": 1e-6
        }
    }
}
model, optimizer, _, _ = deepspeed.initialize(config_params=config, model=model, model_parameters=model.parameters())

# choose tensor backend
BACKEND = "gpu" if torch.cuda.is_available() else "cpu"

# naive local audio dataset
audio_ds = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data', 'sample.mp3'),
    os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'data', 'long_sample.mp3')]

for audio in audio_ds:
    y, _ = librosa.load(audio['data'], sr=16000, mono=True)
    input_values = tokenizer(y, return_tensors="pt", padding="longest").input_values
    decoded = tokenizer.batch_decode(torch.argmax(model(input_values.to(torch.float16).to(BACKEND)).logits, dim=-1))[0]
    print(decoded)
