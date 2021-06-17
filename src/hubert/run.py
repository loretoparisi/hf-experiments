# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os
import sys
import torch
import librosa
from transformers import Wav2Vec2Processor, HubertForCTC

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(BASE_PATH, '..'))
from lpdutils.lpaudioset import LPAudioSet

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft"
    , cache_dir=os.getenv("cache_dir", "../../models"))
model = HubertForCTC.from_pretrained("facebook/hubert-xlarge-ls960-ft"
    , cache_dir=os.getenv("cache_dir", "../../models"))

# naive audio dataset
audio_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'data', 'audio')
audio_ds = [os.path.join(audio_path, 'sample.wav'),
    os.path.join(audio_path, 'long_sample.wav')]


# load local dataset
my_dataset = LPAudioSet(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'data', 'audio'), sr=16000)
train_loader = torch.utils.data.DataLoader(my_dataset, 
                                batch_size=1,
                                shuffle=True, 
                                num_workers=1, 
                                drop_last=True,
                                collate_fn=LPAudioSet.collate_fn)
for idx, audio in enumerate(train_loader):
    print(idx, audio.shape)
    
for audio in audio_ds[0:1]:
    y, _ = librosa.load(audio, sr=16000)
    input_values = processor(y, return_tensors="pt").input_values  # Batch size 1
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])