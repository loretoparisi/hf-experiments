# hf-experiments
Machine Learning Experiments with Hugging Face 🤗

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
│   ├── asr :new:
│   └── summarization
└── wheels
└── models
```

## How to build
To build experiments run
```bash
./build.sh
```

## How to run
To run an experiment run
```bash
./run.sh [experiment_name] [cache_dir_folder]
```


The `experiment_name` field is among the following supported experiment names:

## Experiments
The following experiments are supported
- emotions - emotions detection
- sentiment - sentiment analysis
- summarization - text summarization

and `cache_dir_folder` is the directorty where to cache models files. See later about this.

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

## Dependencies
Dependencies are defined in the `requirements.txt` file and currently are

```bash
torch
transformers
keras
soundfile
```

These will install a number of dependant libraries that can be found in the `install.log`.

## Models files
Where are models files saved? Models files are typically big. It's preferable to save them to a custom folder like an external HDD of a shared disk. For this reason a docker environment variable `cache_dir` can specified at run:

```bash
./run.sh emotions models/
```

the `models` folder will be assigned to the `cache_dir` variable to be used as default alternative location to download pretrained models. A `os.getenv("cache_dir")` will be used to retrieve the environemnt variable in the code.

## Contributors

- [shoegazerstella](https://github.com/shoegazerstella)
