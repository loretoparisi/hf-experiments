#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020-2021 Loreto Parisi (loretoparisi at gmail dot com)

gpu=$1
if [ -z "$gpu" ]; 
then
    echo "Building cpu..."
    docker build -f Dockerfile -t hfexperiments .
else
    echo "Building gpu..."
    docker build -f Dockerfile.gpu -t hfexperimentsgpu .
fi
