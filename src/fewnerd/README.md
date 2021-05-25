# few-nerd
[Few-NERD](https://github.com/thunlp/Few-NERD): Not Only a Few-shot NER Dataset
Few-NERD is a large-scale, fine-grained manually annotated named entity recognition dataset, which contains 8 coarse-grained types, 66 fine-grained types, 188,200 sentences, 491,711 entities and 4,601,223 tokens. Three benchmark tasks are built, one is supervised (Few-NERD (SUP)) and the other two are few-shot (Few-NERD (INTRA) and Few-NERD (INTER))

## Dataset
Download NER datasets
```
./src/fewnerd/dataset.sh supervised
./src/fewnerd/dataset.sh inter
./src/fewnerd/dataset.sh intra
```

This will save the datasets into the `models` folder as defined by the env var `cache_dir`. The models folders loooks like

```
supervised
.
├── test-supervised.txt
├── train-supervised.txt
└── val-supervised.txt
inter
.
├── test-inter.txt
├── train-inter.txt
└── val-inter.txt
intra
.
├── test-intra.txt
├── train-intra.txt
└── val-intra.txt
```

## Train

- 5-way-1~5-shot

```
python src/fewnerd/train.py --train $cache_dir/inter/train-inter.txt \
--val $cache_dir/inter/val-inter.txt --test $cache_dir/inter/test-inter.txt \
--lr 1e-3 --batch_size 2 --trainN 5 --N 5 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 60 --model structshot --tau 0.32
```

- 5-way-5~10-shot

```
python src/fewnerd/train.py --train $cache_dir/inter/train-inter.txt \
--val $cache_dir/inter/val-inter.txt --test $cache_dir/inter/test-inter.txt \
--lr 1e-3 --batch_size 2 --trainN 5 --N 5 --K 5 --Q 5 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 60 --model structshot --tau 0.318
```

- 10-way-1~5-shot

```
python src/fewnerd/train.py --train $cache_dir/inter/train-inter.txt \
--val $cache_dir/inter/val-inter.txt --test $cache_dir/inter/test-inter.txt \
--lr 1e-3 --batch_size 2 --trainN 10 --N 10 --K 1 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 60 --model structshot --tau 0.32
```

- 10-way-5~10-shot

```
python src/fewnerd/train.py --train $cache_dir/inter/train-inter.txt \
--val $cache_dir/inter/val-inter.txt --test $cache_dir/inter/test-inter.txt \
--lr 1e-3 --batch_size 2 --trainN 5 --N 5 --K 5 --Q 1 \
--train_iter 10000 --val_iter 500 --test_iter 5000 --val_step 1000 \
--max_length 60 --model structshot --tau 0.434
```
