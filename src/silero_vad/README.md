# silero-vad
[silero-vad](https://github.com/snakers4/silero-vad)

## How VAD Works
- Audio is split into *250 ms chunks* (you can choose any chunk size, but quality with chunks shorter than 100ms will suffer and there will be more false positives and "unnatural" pauses);
- VAD keeps record of a previous chunk (or zeros at the beginning of the stream);
Then this 500 ms audio (250 ms + 250 ms) is split into N (typically 4 or 8) windows and the model is applied to this window batch. Each window is 250 ms long (naturally, windows overlap);
- Then probability is averaged across these windows;
- Though typically pauses in speech are 300 ms+ or longer (pauses less than 200-300ms are typically not meaninful), it is hard to confidently classify speech vs noise / music on very short chunks (i.e. 30 - 50ms);