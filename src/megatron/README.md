# megatron
[NVIDIA megatron](https://huggingface.co/nvidia)

## models
- [megatron-bert-cased-345m](https://huggingface.co/nvidia/megatron-bert-cased-345m)
- [megatron-gpt2-345m](https://huggingface.co/nvidia/megatron-gpt2-345m)
- [megatron-bert-uncased-345m](https://huggingface.co/nvidia/megatron-bert-uncased-345m)

To download models run `models.sh`:

```
./models.sh
```

models will be saved to `$cache_dir/nvidia/`:

```
models/nvidia
.
├── megatron-bert-cased-345m
│   └── checkpoint.zip
├── megatron-bert-uncased-345m
│   └── checkpoint.zip
└── megatron-gpt2-345m
    └── checkpoint.zip
```

## How to run
To run over `cpu` and save models to default model's cache dir (`./models/`)
```
./run.sh megatron cpu
```

or to run in `debug` mode:

```
./debug.sh
$ python src/megatron/run.py
```