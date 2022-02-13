#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2022 Loreto Parisi (loretoparisi at gmail dot com)

if [ -z "$cache_dir" ]; then
    cache_dir="../../models/"
fi

if [ ! -d $cache_dir ] 
then
    mkdir $cache_dir
fi

echo "Downlading models to $cache_dir..."
cd $cache_dir
curl -O "https://s3.eu-central-1.amazonaws.com/avg-projects/projected_gan/models/pokemon.pkl"
curl -O "https://s3.eu-central-1.amazonaws.com/avg-projects/projected_gan/models/art_painting.pkl"
curl -O "https://s3.eu-central-1.amazonaws.com/avg-projects/projected_gan/models/church.pkl"
curl -O "https://s3.eu-central-1.amazonaws.com/avg-projects/projected_gan/models/cityscapes.pkl"
curl -O "https://s3.eu-central-1.amazonaws.com/avg-projects/projected_gan/models/clevr.pkl"
curl -O "https://s3.eu-central-1.amazonaws.com/avg-projects/projected_gan/models/ffhq.pkl"
curl -O "https://s3.eu-central-1.amazonaws.com/avg-projects/projected_gan/models/flowers.pkl"
curl -O "https://s3.eu-central-1.amazonaws.com/avg-projects/projected_gan/models/landscape.pkl"