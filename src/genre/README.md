# GENRE
FacebookReasearch [GENRE](https://github.com/facebookresearch/GENRE/tree/main/examples_genre) (Generative ENtity REtrieval) for transformers


## Download the dataset
Please use `models.sh` to download models and data or do manually:
[BPE prefix tree (trie) from KILT Wikipedia titles](http://dl.fbaipublicfiles.com/GENRE/kilt_titles_trie_dict.pkl)
```
cd models/
s -l kilt_titles_trie_dict.pkl 
-rw-r--r--@ 1 loretoparisi  staff  215214973 Mar 31 21:46 kilt_titles_trie_dict.pkl
```

## Download the models
Please use `models.sh` to download models and data or do manually:

### Entity Disambiguation
BLINK	[hf_entity_disambiguation_blink](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_blink.tar.gz)
BLINK + AidaYago2	[hf_entity_disambiguation_aidayago](http://dl.fbaipublicfiles.com/GENRE/hf_entity_disambiguation_aidayago.tar.gz)

```
cd models/
tar -xvzf hf_entity_disambiguation_blink.tar.gz
tar -xvzf hf_entity_disambiguation_aidayago.tar.gz
```

### Document Retieval
KILT [hf_wikipage_retrieval](http://dl.fbaipublicfiles.com/GENRE/hf_wikipage_retrieval.tar.gz)
```
cd models/
tar -xvzf hf_wikipage_retrieval.tar.gz
```