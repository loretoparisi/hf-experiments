#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)

if [ -z "$cache_dir" ]; then
    cache_dir="../../models/"
fi

if [ ! -d $cache_dir ] 
then
    mkdir $cache_dir
fi
if [ ! -d "${cache_dir}/nvidia" ] 
then
    mkdir "${cache_dir}/nvidia"
fi
if [ ! -d "${cache_dir}/nvidia/megatron-bert-cased-345m" ] 
then
    mkdir "${cache_dir}/nvidia/megatron-bert-cased-345m"
fi
if [ ! -d "${cache_dir}/nvidia/megatron-gpt2-345m" ] 
then
    mkdir "${cache_dir}/nvidia/megatron-gpt2-345m"
fi
if [ ! -d "${cache_dir}/nvidia/megatron-bert-uncased-345m" ] 
then
    mkdir "${cache_dir}/nvidia/megatron-bert-uncased-345m"
fi

echo "Downlading models to $cache_dir..."
cd $cache_dir
curl --location-trusted --max-redirs 10 -O "https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip"
mv zip nvidia/megatron-bert-cased-345m/checkpoint.zip
curl --location-trusted --max-redirs 10 -O "https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip"
mv zip nvidia/megatron-gpt2-345m/checkpoint.zip
curl --location-trusted --max-redirs 10 -O "https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip"
mv zip nvidia/megatron-bert-uncased-345m/checkpoint.zip


