#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)


haystack='emotions sentiment summarization asr qa genre gpt_neo audioseg colbert luke msmarco mlpvision bigbird silero_vad bert vit nrot'
needle=$1
gpu=$2
cache_dir=$3

if [ -z "$cache_dir" ]; 
then
    cache_dir=$(pwd)/models
fi

if [[ " $haystack " =~ .*\ $needle\ .* ]]; then

    if [ -z "$gpu" ]; 
    then
        echo "Running cpu..."
        docker run -e cache_dir=/app/models -v $cache_dir:"/app/models" --rm -it hfexperiments python s${needle}/run.py
    else
        echo "Running gpu..."
        docker run -e cache_dir=/workspace/app/models -v $cache_dir:"/workspace/app/models" --rm -it --gpus all hfexperimentsgpu python ${needle}/run.py
    fi

else
    echo "experiment ${needle} is not supported yet, please open a new [Feature Request: ${needle}] here https://github.com/loretoparisi/hf-experiments/issues/new"
fi

