# hf-experiments
Machine Learning Experiments with Hugging Face 🤗

## ASR
Automatic Speech Recognition with Wav2vec. Supported pretrained HuggingFace models are `facebook/wav2vec2-large-960h-lv60-self` and `facebook/wav2vec2-base-960h`.

### MP3 to WAV
Please use `convert.sh input_file [duration]` to cut and convert a mp3 input file to mono 16000 Hz wav file.
```
./convert.sh /some_path/sample.mp3
cutting /some_path/sample . mp3 to 30 seconds
converting to wav
wav file: /some_path/sample.wav
```