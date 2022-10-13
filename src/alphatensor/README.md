# AlphaTensor

This is code accompanying the publication

> Fawzi, A. et al. [Discovering faster matrix multiplication algorithms with
reinforcement learning](https://www.nature.com/articles/s41586-022-05172-4).
*Nature* **610** (2022)

There are 4 independent directories:

- `algorithms` contains algorithms discovered by AlphaTensor, represented as
factorizations of matrix multiplication tensors, and a Colab showing how to load
these.

- `benchmarking` contains a script that can be used to measure the actual speed
of matrix multiplication algorithms on an NVIDIA V100 GPU.

- `nonequivalence` contains 14,236 nonequivalent algorithms discovered by
AlphaTensor for the same matrix multiplication problem (multiplying 4x4
matrices), and a Colab that verifies their nonequivalence.

- `recombination` contains the code we used to decompose larger matrix
multiplication tensors by recombining factorizations of smaller ones.

## How to run
```
pip install -r src/alphatensor/requirements.txt
python src/alphatensor/run.py 
```