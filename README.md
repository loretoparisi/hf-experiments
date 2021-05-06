# hf-experiments
Machine Learning Experiments with Hugging Face's (HF) [transformers](https://github.com/huggingface/transformers) 🤗

## What's inside

```bash
.
├── Dockerfile
├── README.md
├── build.sh
├── install.log
├── requirements.txt
├── run.sh
├── src
│   ├── emotions
│   ├── sentiment
│   ├── asr
│   ├── translation
│   ├── genre
│   ├── gpt_neo
│   ├── audioseg
│   ├── colbert
│   ├── luke
│   ├── msmarco
│   └── summarization
└── wheels
└── models
```

## Experiments
The following experiments are supported

- emotions - emotions detection
- sentiment - sentiment analysis
- asr - automatic speech recognition
- translation - text multiple languages translation
- summarization - text summarization
- GENRE - Generative ENtity REtrieval :new:
- gpt_neo - EleutherAI's replication of the GPT-3 :new:
- audioseg - Pyannote audio segmentation and speaker diarization :new:
- colbert - Model is based on ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT :new:
- luke - LUKE is a RoBERTa model that does named entity recognition, extractive and cloze-style question answering, entity typing, and relation classification :new:
- msmarco - Sentence BERT's MSMarco for Semantic Search and Retrieve & Re-Rank :new:

## Libraries
We are up-to-date with the latest `transformers`, `Pytorch`, `tensorflow` and `Keras` models, and we also provide most common ML libraries:

```
Package                 Version     
----------------------- ------------
transformers            4.5.1
tokenizers              0.10.2 
torch                   1.8.1
tensorflow              2.4.1
Keras                   2.4.3
pytorch-lightning       1.2.10
numpy                   1.19.5
tensorboard             2.4.1
sentencepiece           0.1.95
pyannote.core           4.1
librosa                 0.8.0
matplotlib              3.4.1
pandas                  1.2.4 
scikit-learn            0.24.2
scipy                   1.6.3 
```

## How to build
To build experiments run
```bash
./build.sh
```

### How to build GPU
To build experiments with GPU run
```bash
./build.sh gpu
```

## How to run
To run an experiment run
```bash
./run.sh [experiment_name] [gpu|cpu] [cache_dir_folder]
```

## How to run GPU
To run an experiment on GPU run
```bash
./run.sh [experiment_name] gpu [cache_dir_folder]
```

The `experiment_name` field is among the following supported experiment names, while the `cache_dir_folder` parameter is the directorty where to cache models files. See later about this.

## How to debug
To debug the code, without running any experiment
```bash
./debug.sh
root@d2f0e8a5ec76:/app# 
```
This will enter the running image `hfexperiments`. You can now run python scripts manually, like

```
root@d2f0e8a5ec76:/app# python src/asr/run.py
```

NOTE.
For preconfigured experiments, please run the `run.py` script from the main folder `/app`, as the cache directories are following that path, so like `python src/asr/run.py`

### How to debug GPU
To debug for GPU run
```bash
./debug.sh gpu
```

## Dependencies
Glbal Dependencies are defined in the `requirements.txt` file and currently are

```bash
torch
tensorflow
keras
transformers
sentencepiece
soundfile
```

### Dev dependencies
Due to high rate of :new: models pushed to the Huggingface models hub, we provide a `requirements-dev.txt` in order to install the latest `master` branch of `transformers`:

```
./debug.sh
pip install -r requirements-dev.txt
```

### Experiment Dependencies
Experiment level dependencies are specified in app folder `requirements.txt` file like `src/asr/requirements.txt` for `asr` experiment.

## Models files
Where are models files saved? Models files are typically big. It's preferable to save them to a custom folder like an external HDD of a shared disk. For this reason a docker environment variable `cache_dir` can specified at run:

```bash
./run.sh emotions models/
```

the `models` folder will be assigned to the `cache_dir` variable to be used as default alternative location to download pretrained models. A `os.getenv("cache_dir")` will be used to retrieve the environemnt variable in the code.

### Additional models files
Some experiments require additional models to be downloaed, not currently available through Huggingface model's hub, therefore a courtesy download script has been provided in the experiment's folder like, `genre/models.sh` for the following experiments:

- `audioset`
- `genre`

We do not automatically download these files, so please run in debug mode with `debug.sh` and download the models manually, before running those experiments. The download shall be done once, and the models files will be placed in the models' cache folder specified by environment variable `cache_dir` as it happens for the Huggingface's Model Hub.

## Contributors

- [shoegazerstella](https://github.com/shoegazerstella)
