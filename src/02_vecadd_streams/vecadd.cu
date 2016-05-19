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

/*
 * The repeat parameter is added to this example purely to increase the work in
 * the kernel enough so that there can be some non-trivial overlap with
 * asynchronous communication.
 */
__global__ void vector_add(int *C, int *A, int *B, int N, int repeat) {
    int j;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];

        for (j = 0; j < repeat - 1; j++) {
            C[i] += A[i] + B[i];
        }
    }
}

void host_vector_add(int *C, int *A, int *B, int N, int repeat) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
        for (int j = 0; j < repeat - 1; j++) {
            C[i] += A[i] + B[i];
        }
    }
}

int main(int argc, char **argv) {
    int i;
    int N = 1024;
    int ntiles = 4;
    int repeat = 20;
    const int threads_per_block = 128;

    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        ntiles = atoi(argv[2]);
    }
    if (argc > 3) {
        repeat = atoi(argv[3]);
    }

    const int tile_size = (N + ntiles - 1) / ntiles;

    int *A, *B, *C;
    int *d_A, *d_B, *d_C;
    // Allocate space on the host for each array
    CHECK_CUDA(cudaMallocHost((void **)&A, N * sizeof(int)));
    CHECK_CUDA(cudaMallocHost((void **)&B, N * sizeof(int)));
    CHECK_CUDA(cudaMallocHost((void **)&C, N * sizeof(int)));

    // Allocate space on the device for each array
    CHECK_CUDA(cudaMalloc(&d_A, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_B, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_C, N * sizeof(int)));

    // Populate host arrays
    for (i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 2 * i;
        C[i] = 0;
    }

    /*
     * TODO Create 'ntiles' streams objects by malloc-ing enough heap space and
     * using cudaStreamCreate on each.
     */

    const unsigned long long start_device = current_time_ns();

    for (int t = 0; t < ntiles; t++) {
        const int tile_start = t * tile_size;
        int tile_end = (t + 1) * tile_size;
        if (tile_end > N) tile_end = N;
        const int tile_size = tile_end - tile_start;

        /*
         * TODO Convert this to be a cudaMemcpyAsync that uses the t-th stream
         * created above.
         */
        CHECK_CUDA(cudaMemcpy(d_A + tile_start, A + tile_start,
                    tile_size * sizeof(int), cudaMemcpyHostToDevice));
    }

    for (int t = 0; t < ntiles; t++) {
        const int tile_start = t * tile_size;
        int tile_end = (t + 1) * tile_size;
        if (tile_end > N) tile_end = N;
        const int tile_size = tile_end - tile_start;

        /*
         * TODO Convert this to be a cudaMemcpyAsync that uses the t-th stream
         * created above.
         */
        CHECK_CUDA(cudaMemcpy(d_B + tile_start, B + tile_start,
                    tile_size * sizeof(int), cudaMemcpyHostToDevice));
    }

    for (int t = 0; t < ntiles; t++) {
        const int tile_start = t * tile_size;
        int tile_end = (t + 1) * tile_size;
        if (tile_end > N) tile_end = N;
        const int tile_size = tile_end - tile_start;

        /*
         * TODO Convert this to be a cudaMemcpyAsync that uses the t-th stream
         * created above.
         */
        CHECK_CUDA(cudaMemcpy(d_C + tile_start, C + tile_start,
                    tile_size * sizeof(int), cudaMemcpyHostToDevice));
    }

    const int nblocks = (N + threads_per_block - 1) / threads_per_block;
    for (int t = 0; t < ntiles; t++) {
        const int tile_start = t * tile_size;
        int tile_end = (t + 1) * tile_size;
        if (tile_end > N) tile_end = N;
        const int tile_size = tile_end - tile_start;

        /*
         * TODO Add a fourth argument to the <<<...>>> kernel configuration to
         * indicate that this kernel should be run in stream t, ensuring that
         * the asynchronous copies launched above in that stream complete before
         * this kernel works on the data transferred.
         */
        vector_add<<<nblocks, threads_per_block, 0>>>(
                d_C + tile_start, d_B + tile_start, d_A + tile_start, tile_size,
                repeat);
    }

    // Transfer the contents of the output array back to the host
    for (int t = 0; t < ntiles; t++) {
        const int tile_start = t * tile_size;
        int tile_end = (t + 1) * tile_size;
        if (tile_end > N) tile_end = N;
        const int tile_size = tile_end - tile_start;

        /*
         * TODO Convert this to be a cudaMemcpyAsync that uses the t-th stream
         * created above.
         */
        CHECK_CUDA(cudaMemcpy(C + tile_start, d_C + tile_start,
                    tile_size * sizeof(int), cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    const unsigned long long end_device = current_time_ns();

    // Validate GPU results
    for (i = 0; i < N; i++) {
        if (C[i] != repeat * (A[i] + B[i])) {
            fprintf(stderr, "Mismatch at index %d: expected %d but got %d\n", i,
                    repeat * (A[i] + B[i]), C[i]);
            return 1;
        }
    }

    // Run on the host
    const unsigned long long start_host = current_time_ns();
    host_vector_add(C, B, A, N, repeat);
    const unsigned long long end_host = current_time_ns();

    const unsigned long long elapsed_device = (end_device - start_device) / 1000;
    const unsigned long long elapsed_host = (end_host - start_host) / 1000;

    printf("Finished! All %d elements validate using %d threads per block.\n",
            N, threads_per_block);
    printf("Took %llu microseconds on the host\n", elapsed_host);
    printf("Took %llu microseconds on the device, %2.5fx speedup\n",
            elapsed_device, (double)elapsed_host / (double)elapsed_device);

    return 0;
}
