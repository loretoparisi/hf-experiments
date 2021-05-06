# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)
# HF: https://huggingface.co/facebook/wav2vec2-base-960h?s=09

from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import torch
import librosa
import os

# facebook/wav2vec2-large-960h-lv60-self
# facebook/wav2vec2-large-xlsr-53-italian

# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained(
    "facebook/wav2vec2-base-960h", cache_dir=os.getenv("cache_dir", "../../models"))
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h", cache_dir=os.getenv("cache_dir", "../../models"))

def trace_mem(nframe=6,top=8):
    '''
        naive memory trace
    '''
    import tracemalloc
    is_tracing = tracemalloc.is_tracing()
    if not is_tracing:
        # start tracing
        tracemalloc.start(nframe)
        return {}
    else:
        # stop tracing
        tracemalloc.stop()
        # read traced memory alloc
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        overhead = tracemalloc.get_tracemalloc_memory()
        stats = tracemalloc.take_snapshot().statistics('traceback')[:top]
        # memory summary
        summary = {}
        summary['memory'] = int(current_mem // 1024)
        summary['peak'] = int(peak_mem // 1024)
        summary['overhead'] = int(overhead // 1024)
        summary['description'] = "traced memory: %d KiB  peak: %d KiB  overhead: %d KiB" % (
            int(current_mem // 1024), int(peak_mem // 1024), int(overhead // 1024)
        )

        # traceback
        out_lines = []
        for trace in stats:
            stacktrace = {}
            stacktrace['memory'] = int(trace.size // 1024)
            stacktrace['blocks'] = int(trace.count)
            stacktrace['stack'] = trace.traceback.format()
            out_lines.append(stacktrace)
        
        data = {}
        data['summary'] = summary
        data['traceback'] = out_lines

        return data

# toy audio dataset
audio_ds = [os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data', 'sample.mp3'),
    os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'data', 'long_sample.mp3')]

for audio in audio_ds[0:1]:
    
    # load audio file
    y, _ = librosa.load(audio, sr=16000, mono=True)

    trace_mem(nframe=6, top=8)

    # tokenize audio
    input_values = tokenizer(y, return_tensors="pt", padding="longest").input_values  # Batch size 1
    # retrieve logits
    logits = model(input_values).logits
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    t = tokenizer.batch_decode(predicted_ids)
    
    print("\n",t,"\n")
    print("\n--------mem--------\n", trace_mem(nframe=6, top=8), "\n--------mem--------")