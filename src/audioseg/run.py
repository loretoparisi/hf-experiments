# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)
# HF: https://huggingface.co/pyannote/segmentation

import os

# pyannote
from pyannote.audio import Model
from pyannote.audio import Inference
from pyannote.audio.pipelines import Segmentation
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines import OverlappedSpeechDetection

# cache_dir supported on develop branch of pyannote only
cache_dir=os.getenv("cache_dir", "../../models")
# if you are using non develop branch of pyannote
os.environ['PYANNOTE_CACHE'] = cache_dir

# naive audio dataset
audio_ds = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data', 'sample.wav'),
    os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data', 'long_sample.wav')]

model = Model.from_pretrained("pyannote/segmentation", cache_dir=cache_dir)
inference = Inference(model)
segmentation = inference(audio_ds[0])
# `segmentation` is a pyannote.core.SlidingWindowFeature
# instance containing raw segmentation scores

pipeline = Segmentation(segmentation=model)
HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speaker turn shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill within speaker pauses shorter than that many seconds.
  "min_duration_off": 0.0
}

pipeline.instantiate(HYPER_PARAMETERS)
segmentation = pipeline(audio_ds[0])
# `segmentation` now is a pyannote.core.Annotation
# instance containing a hard binary segmentation 
print(segmentation)

# Voice activity detection
pipeline = VoiceActivityDetection(segmentation=model)
pipeline.instantiate(HYPER_PARAMETERS)
vad = pipeline(audio_ds[0])
print(vad)

# Overlapped speech detection
pipeline = OverlappedSpeechDetection(segmentation=model)
pipeline.instantiate(HYPER_PARAMETERS)
osd = pipeline(audio_ds[0])
print(osd)