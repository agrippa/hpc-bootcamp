/*
 * MIT License
 * 
 * Copyright (c) 2016, Max Grossman, Computer Science Department, Rice University
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <stdio.h>
#include <assert.h>
#include <common.h>

__global__ void vector_add(int *C, int *A, int *B, int N) {
    int i;
    for (i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv) {
    int i;
    int N = 1024;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    int *A, *B, *C;
    CHECK_CUDA(cudaMallocManaged(&A, N * sizeof(int)));
    CHECK_CUDA(cudaMallocManaged(&B, N * sizeof(int)));
    CHECK_CUDA(cudaMallocManaged(&C, N * sizeof(int)));

    // Populate arrays
    for (i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    vector_add<<<1, 1>>>(C, B, A, N);

    CHECK_CUDA(cudaDeviceSynchronize());

    // Validate results
    for (i = 0; i < N; i++) {
        assert(C[i] == A[i] + B[i]);
    }

    return 0;
}
