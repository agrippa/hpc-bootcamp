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
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

void host_vector_add(int *C, int *A, int *B, int N) {
    for (int i = 0; i < N; i++) {
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
    int *d_A, *d_B, *d_C;
    // Allocate space on the host for each array
    A = (int *)malloc(N * sizeof(int)); assert(A);
    B = (int *)malloc(N * sizeof(int)); assert(B);
    C = (int *)malloc(N * sizeof(int)); assert(C);

    // Allocate space on the device for each array
    CHECK_CUDA(cudaMalloc(&d_A, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_B, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_C, N * sizeof(int)));

    // Populate host arrays
    for (i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
    }

    const unsigned long long start_device = current_time_ns();
    // Transfer the contents of the input host arrays on to the device
    CHECK_CUDA(cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice));

    vector_add<<<1, 1>>>(d_C, d_B, d_A, N);

    // Transfer the contents of the output array back to the host
    CHECK_CUDA(cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost));
    const unsigned long long end_device = current_time_ns();

    // Validate GPU results
    for (i = 0; i < N; i++) {
        assert(C[i] == A[i] + B[i]);
    }

    // Run on the host
    const unsigned long long start_host = current_time_ns();
    host_vector_add(C, B, A, N);
    const unsigned long long end_host = current_time_ns();

    const unsigned long long elapsed_device = (end_device - start_device) / 1000;
    const unsigned long long elapsed_host = (end_host - start_host) / 1000;

    printf("Finished! All %d elements validate.\n", N);
    printf("Took %llu microseconds on the host\n", elapsed_host);
    printf("Took %llu microseconds on the device, %2.5fx speedup\n", elapsed_device, (double)elapsed_host / (double)elapsed_device);

    return 0;
}
