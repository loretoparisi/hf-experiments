#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020 Loreto Parisi (loretoparisi at gmail dot com)


haystack='emotions sentiment summarization asr'
gpu=$1
cache_dir=$2

if [ -z "$cache_dir" ]; then
    cache_dir=models
fi

if [ -z "$gpu" ]; 
then
    echo "Running cpu..."
    docker run -e cache_dir=$cache_dir -v $(pwd):/app --rm -it hfexperiments bash
else
    echo "Running gpu..."
    docker run -e cache_dir=$cache_dir -v $(pwd):/app --rm -it --gpus all hfexperimentsgpu bash
fi


