#!/bin/bash
#
# @author Loreto Parisi (loretoparisi at gmail dot com)
# Copyright (c) 2021 Loreto Parisi (loretoparisi at gmail dot com)
echo "Downlading models..."
cd ../models/
curl -O http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_blink.tar.gz
curl -O http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_aidayago.tar.gz
tar -xvzf hf_entity_disambiguation_blink.tar.gz
tar -xvzf hf_entity_disambiguation_aidayago.tar.gz
cd -
echo "Downlading dataset..."
cd ../models/
curl -O http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie_dict.pkl
cd -