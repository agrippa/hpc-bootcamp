#!/usr/bin/python
import sys
import numpy as np

def kernel(nxt, curr, N):
    for i in range(1, N + 1):
        nxt[i] = (curr[i - 1] + curr[i + 1]) / 2.0


def driver(niters, seed):
    curr = seed
    nxt = np.zeros(len(seed))
    nxt[0] = seed[0]
    nxt[-1] = seed[-1]

    for iter in range(niters):
        kernel(nxt, curr, len(curr) - 2)

        tmp = nxt
        nxt = curr
        curr = tmp

    return curr

if len(sys.argv) != 3:
    sys.stderr('usage: python 1d_iter_avg.py <N> <niters>\n')
    sys.exit(1)

N = int(sys.argv[1])
niters = int(sys.argv[2])

seed = np.zeros(N + 2)
seed[-1] = 1.0
result = driver(niters, seed)

from matplotlib import pyplot as plt
plt.imshow(result, interpolation='nearest')
plt.show()
