#!/usr/bin/python
import sys
import time
import math
import numpy as np
import matplotlib
matplotlib.use('agg')

from numba import cuda

@cuda.jit
def kernel(nxt, curr, N):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if tid < N:
        tid += 1
        nxt[tid] = (curr[tid - 1]  + curr[tid + 1]) / 2.0


def driver(niters, seed):
    curr = seed
    nxt = np.zeros(len(seed))
    nxt[0] = seed[0]
    nxt[-1] = seed[-1]

    start_time = time.time()

    threads_per_block = 256
    blocks_per_grid = math.ceil((len(curr) - 2) / threads_per_block)

    # TODO 1. Transfer the nxt and curr arrays to the CUDA device.
    for iter in range(niters):
        # TODO 2. Change the arrays passed here to be device arrays rather than
        # host arrays.
        kernel[blocks_per_grid, threads_per_block](nxt, curr, len(curr) - 2)

        # TODO 3. Swap device arrays instead of host arrays
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
