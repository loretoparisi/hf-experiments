# AlphaTensor

This is code accompanying the publication

> Fawzi, A. et al. [Discovering faster matrix multiplication algorithms with
reinforcement learning](https://www.nature.com/articles/s41586-022-05172-4).
*Nature* **610** (2022)

## Code

- `algorithms` contains algorithms discovered by AlphaTensor, represented as
factorizations of matrix multiplication tensors, and a Colab showing how to load
these.

## How to run
```
pip install -r src/alphatensor/requirements.txt
python src/alphatensor/run.py 
```