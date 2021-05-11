# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference demo for YAMNet."""
from __future__ import division, print_function

import sys, os
import json, codecs
import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf

import params as yamnet_params
import yamnet as yamnet_model


def main(argv):
  assert argv, 'Usage: inference.py <wav file> <wav file> ...'

  model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yamnet.h5')
  classes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yamnet_class_map.csv')
  event_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'event.json')
  
  params = yamnet_params.Params()
  yamnet = yamnet_model.yamnet_frames_model(params)
  yamnet.load_weights(model_path)
  yamnet_classes = yamnet_model.class_names(classes_path)

  for file_name in argv:
    # Decode the WAV file.
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
      waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
      waveform = resampy.resample(waveform, sr, params.sample_rate)

    # Predict YAMNet classes.
    scores, embeddings, spectrogram = yamnet(waveform)
    # Scores is a matrix of (time_frames, num_classes) classifier scores.
    # Average them along time to get an overall classifier output for the clip.
    prediction = np.mean(scores, axis=0)
    # Report the highest-scoring classes and their scores.
    top5_i = np.argsort(prediction)[::-1][:5]
    print(file_name, ':\n' +
          '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                    for i in top5_i))
    
    # print all classes
    b = prediction.tolist() # nested lists with same data, indices
    pred = []
    for (i,cls) in enumerate(yamnet_classes):
      item={}
      item['label']=cls
      item['value']=round(b[i], 6)
      pred.append(item)
    pred = sorted(pred, key=lambda x: x['value'], reverse=True)
    json.dump(pred, codecs.open(event_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

if __name__ == '__main__':
  main(sys.argv[1:])
