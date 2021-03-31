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
Please use `models.sh` to download models and data or do manually to the `models` folder:

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

### End-to-End Entity Linking
WIKIPEDIA	[hf_e2e_entity_linking_wiki_abs](http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_wiki_abs.tar.gz)
WIKIPEDIA + AidaYago2	[hf_e2e_entity_linking_aidayago](http://dl.fbaipublicfiles.com/GENRE/hf_e2e_entity_linking_aidayago.tar.gz)
```
cd models/
tar -xvzf hf_e2e_entity_linking_wiki_abs.tar.gz
tar -xvzf hf_e2e_entity_linking_aidayago.tar.gz
```

## How to Run
Please be sure to have installed additional requirements in `requirements.txt`. 
Run the experiment by name with `run.sh` script as usual
```
./run.sh genre
```

or entering debug mode with `debug.sh` and then manually calling the script:

```
./debug.sh
$ pip install -r src/genre/requirements.txt
$ python src/genre/run.py
```

You should get the following output
```
[{'text': 'Germany', 'logprob': tensor(-0.1856)}, {'text': 'German Empire', 'logprob': tensor(-2.1858)}, {'text': 'Nazi Germany', 'logprob': tensor(-2.4682)}, {'text': 'German language', 'logprob': tensor(-3.2784)}, {'text': 'France', 'logprob': tensor(-4.2070)}]

[{'text': 'Albert Einstein', 'logprob': tensor(-0.0708)}, {'text': 'Erwin Schrödinger', 'logprob': tensor(-1.3913)}, {'text': 'Werner Bruschke', 'logprob': tensor(-1.5358)}, {'text': 'Werner von Habsburg', 'logprob': tensor(-1.8696)}, {'text': 'Werner von Moltke', 'logprob': tensor(-2.2318)}]

```