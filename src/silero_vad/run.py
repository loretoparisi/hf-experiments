# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

import os

import torch
from pprint import pprint

torch.set_num_threads(1)

# pytorch models hub dir to 'cache_dir'
cache_dir=os.getenv("cache_dir", "../../models")
torch.hub.set_dir(cache_dir)

# check model exists in cache dir
force_reload = os.path.isdir(os.path.join(cache_dir, 'snakers4/silero-vad'))

# Example 1: Voice Activity Detector
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=force_reload)

(get_speech_ts,
 get_speech_ts_adaptive,
 _, read_audio,
 _, _, _) = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

wav = read_audio(f'{files_dir}/en.wav')
# full audio
# get speech timestamps from full audio file

# classic way
speech_timestamps = get_speech_ts(wav, model,
                                  num_steps=4)
print("speech timestamps")
pprint(speech_timestamps)

# adaptive way
speech_timestamps = get_speech_ts_adaptive(wav, model)

print("speech timestamps adaptive")
pprint(speech_timestamps)


# Example 2: Number Detector
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_number_detector',
                              force_reload=False)

(get_number_ts,
 _, read_audio,
 _, _) = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

wav = read_audio(f'{files_dir}/en_num.wav')
# full audio
# get number timestamps from full audio file
number_timestamps = get_number_ts(wav, model)

print("number timestamps")
pprint(number_timestamps)

# Example 3: Language Classifier
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_lang_detector',
                              force_reload=False)

get_language, read_audio = utils

files_dir = torch.hub.get_dir() + '/snakers4_silero-vad_master/files'

wav = read_audio(f'{files_dir}/de.wav')
language = get_language(wav, model)

print("language")
pprint(language)