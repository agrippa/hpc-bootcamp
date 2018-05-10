#!/usr/bin/python
import sys
import time
import math
import numpy as np
import matplotlib
matplotlib.use('agg')

# TODO 1. Add an import for the numba cuda package

# TODO 2. Decorate kernel to indicate to numba that it should be offloaded to a
# CUDA device.
def kernel(nxt, curr, N):
    for i in range(1, N + 1):
        nxt[i] = (curr[i - 1] + curr[i + 1]) / 2.0


def driver(niters, seed):
    curr = seed
    nxt = np.zeros(len(seed))
    nxt[0] = seed[0]
    nxt[-1] = seed[-1]

    start_time = time.time()
    for iter in range(niters):
        # TODO 3. Select a threads per block and blocks per grid, then use these
        # values to spawn a CUDA kernel using numba.
        kernel(nxt, curr, len(curr) - 2)

        tmp = nxt
        nxt = curr
        curr = tmp
    elapsed_time = time.time() - start_time

    print('Elapsed time for N=' + str(len(seed) - 2) + ', # iters=' +
            str(niters) + ' is ' + str(elapsed_time) + ' s')
    print(str(float(niters) / elapsed_time) + ' iters / s')

    return curr

if len(sys.argv) != 3:
    sys.stderr.write('usage: python 1d_iter_avg.py <N> <niters>\n')
    sys.exit(1)

N = int(sys.argv[1])
niters = int(sys.argv[2])

seed = np.zeros(N + 2)
seed[-1] = 1.0
result = driver(niters, seed)

# Save to image
from matplotlib import pyplot as plt
ny = int(len(result) / 10)
if ny < 1:
    ny = 1
img = np.zeros((ny, len(result)))
for i in range(img.shape[0]):
    img[i, :] = result
plt.imshow(img, interpolation='nearest')
plt.savefig('img.png')
