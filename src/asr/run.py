# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)
# HF: https://huggingface.co/facebook/wav2vec2-base-960h?s=09

from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForMaskedLM, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch
import os

# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self",
    cache_dir=os.getenv("cache_dir", "model"))
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self",
    cache_dir=os.getenv("cache_dir", "model"))

# load model and tokenizer
#tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h",cache_dir=os.getenv("cache_dir", "model"))
#model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h",cache_dir=os.getenv("cache_dir", "model"))

# define function to read in sound file
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    print( batch["speech"].shape )
    return batch

# define function to read in sound file
def map_files_to_array(file_path):
    batch = {}
    speech, _ = sf.read(file_path)
    batch["speech"] = speech
    print( batch["speech"].shape )
    return batch

# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)
batch_of_waveforms = ds["speech"][:2]

# this is the waveform input file
ds_file = [ os.path.join(os.path.dirname(os.path.abspath(__file__)),'data', 'sample.wav') ]
batch_of_waveforms_file = [ map_files_to_array(d) for d in ds_file ]

# tokenize
input_values = tokenizer(batch_of_waveforms, return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)
print(transcription)
