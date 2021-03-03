#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020 Loreto Parisi (loretoparisi at gmail dot com)


haystack='emotions sentiment summarization asr qa'
needle=$1
cache_dir=$2

if [ -z "$cache_dir" ]; then
    cache_dir=models
fi

if [[ " $haystack " =~ .*\ $needle\ .* ]]; then
    docker run -e cache_dir=$cache_dir -v $(pwd):/app --rm -it hfexperiments python src/${needle}/run.py
else
    echo "experiment ${needle} is not supported yet, please open a new [Feature Request: ${needle}] here https://github.com/loretoparisi/hf-experiments/issues/new"
fi

