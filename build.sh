#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020 Loreto Parisi (loretoparisi at gmail dot com)

# check wheels
cd wheels
if [ -f "tensorflow-2.2.0-cp37-cp37m-manylinux2010_x86_64.whl" ]; then
    echo "tensorflow-2.2.0-cp37-cp37m-manylinux2010_x86_64.whl"
else
    wget https://files.pythonhosted.org/packages/4c/1a/0d79814736cfecc825ab8094b39648cc9c46af7af1bae839928acb73b4dd/tensorflow-2.2.0-cp37-cp37m-manylinux2010_x86_64.whl
fi
if [ -f "torch-1.5.0-cp37-cp37m-manylinux1_x86_64.whl" ]; then
    echo "torch-1.5.0-cp37-cp37m-manylinux1_x86_64.whl"
else
    wget https://files.pythonhosted.org/packages/76/58/668ffb25215b3f8231a550a227be7f905f514859c70a65ca59d28f9b7f60/torch-1.5.0-cp37-cp37m-manylinux1_x86_64.whl
fi
cd ..

docker build -f Dockerfile -t hfexperiments .
