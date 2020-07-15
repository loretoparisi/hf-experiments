#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020 Loreto Parisi (loretoparisi at gmail dot com)


haystack='emotions sentiment summarization'
needle=$1
cache_dir=$2

if [ -z "$cache_dir" ]; then
    cache_dir=model
fi

if [[ " $haystack " =~ .*\ $needle\ .* ]]; then
    docker run -e cache_dir=$cache_dir -v $(pwd):/app --rm -it hfexperiments python src/$1/run.py
else
    echo "experiment $1 is not supported, please open a new issue here https://github.com/loretoparisi/hf-experiments/issues/new"
fi

