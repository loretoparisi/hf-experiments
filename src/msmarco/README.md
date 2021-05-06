# msmarco
SentenceBERT msmarco for [Retrieve & Re-Rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) and [Semantic Search](https://www.sbert.net/examples/applications/semantic-search/README.html).

## How to run
To run over `cpu` and save models to default model's cache dir (`./models/`)
```
./run.sh msmarco cpu
```

or to run in `debug` mode:

```
./debug.sh
$ python src/msmarco/run.py
```

## Examples
```
query_embeddings: torch.Size([2, 384])
passage_embeddings: torch.Size([2, 384])
cosine_similarity: tensor([0.8791])
QUESTION: What is the capital of France?
        PASSAGE: Paris is the capital of France  (score: 0.8791)
        PASSAGE: New York City is the most populous city in the United States, with an estimated 8,336,817 people living in the city, according to U.S. Census estimates dating July 1, 2019  (score: 0.1299)
QUESTION: How many people live in New York City?
        PASSAGE: New York City is the most populous city in the United States, with an estimated 8,336,817 people living in the city, according to U.S. Census estimates dating July 1, 2019  (score: 0.8416)
        PASSAGE: Paris is the capital of France  (score: 0.0697)
```