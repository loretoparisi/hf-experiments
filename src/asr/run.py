# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)

from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForMaskedLM
import soundfile as sf
import torch
import os

# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h",
    cache_dir=os.getenv("cache_dir", "model"))
model = Wav2Vec2ForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h",
    cache_dir=os.getenv("cache_dir", "model"))

# define function to read in sound file
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

# tokenize
soundfile_path = 'data/sample.mp3'
input_values = tokenizer(soundfile_path, return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)