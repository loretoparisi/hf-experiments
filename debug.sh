#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020 Loreto Parisi (loretoparisi at gmail dot com)


haystack='emotions sentiment summarization asr'
cache_dir=$1

if [ -z "$cache_dir" ]; then
    cache_dir=models
fi

docker run -e cache_dir=$cache_dir -v $(pwd):/app --rm -it hfexperiments bash


