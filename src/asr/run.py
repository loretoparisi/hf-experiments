# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)
# HF: https://huggingface.co/facebook/wav2vec2-base-960h?s=09

from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import torch
import librosa
import os
import json
from segmenter import Segmenter

# load model and tokenizer
#tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", cache_dir=os.getenv("cache_dir", "../models"))
#model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self", cache_dir=os.getenv("cache_dir", "../models"))

# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained(
    "facebook/wav2vec2-base-960h", cache_dir=os.getenv("cache_dir", "../../models"))
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h", cache_dir=os.getenv("cache_dir", "../../models"))

audio_ds = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data', 'sample.mp3'),
    os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'data', 'long_sample.mp3')]

# create speech segmenter
seg = Segmenter(model_path=os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'speech_segmenter_models'),
    vad_engine='smn', detect_gender=True,
    ffmpeg='ffmpeg', batch_size=32)

# it holds audio segmentations
segmentations = []
for audio in audio_ds:
    # [('noEnergy', 0.0, 0.8), ('male', 0.8, 9.84), ('music', 9.84, 10.96), ('male', 10.96, 14.98)]
    # segmentation = seg(audio, start_sec=0, stop_sec=30)
    s = seg(audio)

    res = {}
    res['segmentation'] = s
    res['audio'] = audio
    segmentations.append(res)

# it holds speech transcriptions
transcriptions = []
for segmentation in segmentations:

    result = {}
    result['audio'] = segmentation['audio']

    # it holds audio speech waveforms
    speech = []
    for s in segmentation['segmentation']:

        start = round(s[1], 3)
        end = round(s[2], 3)
        duration = round(end-start, 3)

        res = {}
        res['start'] = start
        res['end'] = end
        res['duration'] = duration
        res['label'] = s[0]

        # skip no voices
        if s[0] == 'male' or s[0] == 'female':
            y, _ = librosa.load(segmentation['audio'], sr=16000,
                                mono=True, offset=start, duration=duration)
            res['speech'] = y

        speech.append(res)

    # it holds speech transcriptions
    transcription = []
    for s in speech:

        res = {}
        res['start'] = s['start']
        res['end'] = s['end']
        res['duration'] = s['duration']
        res['label'] = s['label']

        if s['label'] == 'male' or s['label'] == 'female':
            # tokenize
            input_values = tokenizer(
                s['speech'], return_tensors="pt", padding="longest").input_values  # Batch size 1
            # retrieve logits
            logits = model(input_values).logits
            # take argmax and decode
            predicted_ids = torch.argmax(logits, dim=-1)
            t = tokenizer.batch_decode(predicted_ids)
            res['transcription'] = t

        transcription.append(res)

    result['transcription'] = transcription

    transcriptions.append(result)

print(json.dumps(transcriptions))
