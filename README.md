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
│   └── summarization
└── wheel
└── model
```

## How to build
To build experiments run
```bash
./build.sh
```

## How to run
To run an experiment run
```bash
./run.sh [experiment_name]
```

where `experiment_name` is among the following supported experiment names:

## Experiments
The following experiments are supported
- emotions - emotions detection
- sentiment - sentiment analysis
- summarization - text summarization


## Depencendies
Dependecies are defined in the `requirements.txt` file and currently are

```bash
tensorflow==2.2.0
torch==1.5.0
transformers==3.0.2
```
These will install a number of dependant libraries that can be found in the `install.log`.

## Wheel? What's that?
I'm using install from local wheels if avaiable. This will speed up build and tests, avoding to transfer several times data over the internet:

```bash
Collecting torch==1.5.0
  Downloading https://files.pythonhosted.org/packages/76/58/668ffb25215b3f8231a550a227be7f905f514859c70a65ca59d28f9b7f60/torch-1.5.0-cp37-cp37m-manylinux1_x86_64.whl (752.0MB)
```
  
I download once the big wheels for `pytorch` (752 MB) and `tensorflow` ((516.2 MB) in the `wheel` folder and check for them before building:

```bash
└── wheel
    ├── tensorflow-2.2.0-cp37-cp37m-manylinux2010_x86_64.whl
    └── torch-1.5.0-cp37-cp37m-manylinux1_x86_64.whl
```

Check the downloadable wheels here

- tensorflow, https://pypi.org/project/tensorflow/#files
- pytorch, https://pypi.org/project/torch/#files 
