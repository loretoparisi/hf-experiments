#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2020 Loreto Parisi (loretoparisi at gmail dot com)
echo "Downlading YamNet models..."
cd yamnet/
curl -O https://storage.googleapis.com/audioset/yamnet.h5
cd ..
echo "Downlading VGGish models..."
cd vggish/
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
cd ..