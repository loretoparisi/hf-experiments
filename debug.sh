#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020 Loreto Parisi (loretoparisi at gmail dot com)


gpu=$1
cache_dir=$2

if [ -z "$cache_dir" ]; then
    cache_dir=$(pwd)/models
fi

if [ -z "$gpu" ]; 
then
    echo "Running cpu..."
    docker run -e cache_dir=/app/models -v $cache_dir:"/app/models" -v $(pwd)/src:/app --rm -it hfexperiments bash
else
    echo "Running gpu..."
    docker run -e cache_dir=/workspace/app/models -v $cache_dir:"/workspace/app/models" -v $(pwd)/src:/workspace/app --rm -it --gpus all hfexperimentsgpu bash
fi