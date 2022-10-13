# hf-experiments
# @author Loreto Parisi (loretoparisi at gmail dot com)
# @see https://www.deepmind.com/blog/discovering-novel-algorithms-with-alphatensor
# Copyright (c) 2022 Loreto Parisi (loretoparisi at gmail dot com)

import numpy as np
import os

def get_mamu_tensor_rectangular(a: int, b: int, c: int) -> np.ndarray:
  """Returns the symmetrized matrix multiplication tensor T_{a, b, c}."""
  result = np.full((a*b, b*c, c*a), 0, dtype=np.int32)
  for i in range(a):
    for j in range(b):
      for k in range(c):
        result[i * b  + j][j * c + k][k * a + i] = 1
  return result

# factorizations_r.npz: algorithms in standard arithmetic
# factorizations_f2.npz: algorithms in arithmetic modulo 2
algos = [ 'factorizations_r.npz', 'factorizations_f2.npz' ]
for algo in algos:

    factorizations = dict(np.load(os.path.join('alphatensor',os.path.dirname(
    os.path.abspath(__file__)),f'algorithms/{algo}'), allow_pickle=True))

    # Print available factorizations and their shapes.
    for key in factorizations:
        u, v, w = factorizations[key]
        rank = u.shape[-1]
        assert rank == v.shape[-1] and rank == w.shape[-1]
        print(f'{key}: rank={u.shape[-1]}')

    # Test correctness of a factorization.
    tensor = get_mamu_tensor_rectangular(3, 4, 5)
    u, v, w = factorizations['3,4,5']
    reconstruction = np.einsum('ir,jr,kr->ijk', u, v, w)
    if np.array_equal(tensor, reconstruction):
        print('Factorization is correct in R (standard arithmetic).')
    elif np.array_equal(tensor, np.mod(reconstruction, 2)):
        print('Factorization is correct in F2 (modular arithmetic).')
    else:
        print('Factorization is incorrect.')