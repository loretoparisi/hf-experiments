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

### Entity Disambiguation and Document Retieval
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

### Example: End-to-End Entity Linking
```
$ python src/genre/entitylinking.py
[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ].', 'logprob': tensor(-0.9068)}, {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Physiology ].', 'logprob': tensor(-1.0778)}, {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ] {. } [ Ernest Einstein ]', 'logprob': tensor(-1.1164)}, {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ] {. } [ Einstein ]', 'logprob': tensor(-1.1661)}, {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physiology or Medicine ] {. } [ Max Einstein ]', 'logprob': tensor(-1.1712)}]

[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a Nobel Prize.', 'logprob': tensor(-1.4009)}, {'text': 'In 1921, { Einstein } [ Einstein (crater) ] received a Nobel Prize.', 'logprob': tensor(-1.6665)}, {'text': 'In 1921, { Einstein } [ Albert Albert Einstein ] received a Nobel Prize.', 'logprob': tensor(-1.7498)}, {'text': 'In 1921, { Einstein } [ Ernest Einstein ] received a Nobel Prize.', 'logprob': tensor(-1.8327)}, {'text': 'In 1921, { Einstein } [ Max Einstein ] received a Nobel Prize.', 'logprob': tensor(-1.8757)}]

[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ].', 'logprob': tensor(-0.8925)}, {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize. } [ Nobel Prize in Physics ]', 'logprob': tensor(-0.8990)}, {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel Prize } [ Nobel Prize in Physics ] {. } [ NIL ]', 'logprob': tensor(-1.7828)}, {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize } [ NIL ].', 'logprob': tensor(-1.8835)}, {'text': 'In 1921, { Einstein } [ Albert Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize. } [ NIL ]', 'logprob': tensor(-1.9888)}]

[{'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize.', 'logprob': tensor(-1.5417)}, {'text': 'In 1921, { Einstein } [ Einstein ] received a Nobel Prize.', 'logprob': tensor(-2.1319)}, {'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize } [ NIL ].', 'logprob': tensor(-2.2816)}, {'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] { Prize. } [ NIL ]', 'logprob': tensor(-2.3914)}, {'text': 'In 1921, { Einstein } [ Einstein ] received a { Nobel } [ Nobel Prize in Physics ] Prize {. } [ NIL ]', 'logprob': tensor(-2.6675)}]

[{'text': 'In 1921, { Einstein } [ Albert Einstein ] received a Nobel Prize.', 'logprob': tensor(-1.4009)}, {'text': 'In 1921, Einstein received a { Nobel Prize } [ Nobel Prize in Physics ].', 'logprob': tensor(-1.8266)}, {'text': 'In 1921, Einstein received a { Nobel Prize } [ Nobel Prize in Medicine ].', 'logprob': tensor(-2.2954)}, {'text': 'In 1921, Einstein received a Nobel Prize.', 'logprob': tensor(-3.4495)}, {'text': 'In 1921, Einstein received a Nobel Prize.', 'logprob': tensor(-100000000.)}]

In 1921, [Einstein](https://en.wikipedia.org/wiki/Albert_Einstein) received a Nobel Prize.

[[(9, 8, 'Albert_Einstein')]]
```

### Custom End-to-End Entity Linking evaluation

```
$ python src/genre/evaluation.py 
micro_p=0.2500 micro_r=0.4000, micro_f1=0.3077, macro_p=0.2500, macro_r=0.4167, macro_f1=0.3095
```