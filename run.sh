#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)


haystack='emotions sentiment summarization asr qa genre gpt_neo audioseg colbert luke msmarco mlpvision bigbird silero_vad'
needle=$1
gpu=$2
cache_dir=$3

if [ -z "$cache_dir" ]; 
then
    cache_dir=models
fi

if [[ " $haystack " =~ .*\ $needle\ .* ]]; then

    if [ -z "$gpu" ]; 
    then
        echo "Running cpu..."
        docker run -e cache_dir=$cache_dir -v $cache_dir:"/${cache_dir}" -v $(pwd):/app --rm -it hfexperiments python src/${needle}/run.py
    else
        echo "Running gpu..."
        docker run -e cache_dir=$cache_dir -v $cache_dir:"/${cache_dir}" -v $(pwd):/app --rm -it --gpus all hfexperimentsgpu python src/${needle}/run.py
    fi

else
    echo "experiment ${needle} is not supported yet, please open a new [Feature Request: ${needle}] here https://github.com/loretoparisi/hf-experiments/issues/new"
fi

