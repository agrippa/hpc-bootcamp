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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vector_add_read_offset(int *C, int *A, int *B, int N, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = i + offset;

    if (k < N) {
        C[i] = A[k] + B[k];
    }
}

__global__ void vector_add_write_offset(int *C, int *A, int *B, int N, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = i + offset;

    if (k < N) {
        C[k] = A[i] + B[i];
    }
}

__global__ void vector_add_weirdly_coalesced(int *C, int *A, int *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        const int warp_id = (i / 32); // int floor
        const int warp_data_offset = warp_id * 32;
        const int warp_data_length = N - warp_data_offset >= 32 ? 32 : N - warp_data_offset;
        const int half_warp_data_length = warp_data_length / 2;
        int offset_in_warp = (i - warp_data_offset) % warp_data_length;
        if (offset_in_warp < half_warp_data_length) {
            offset_in_warp = half_warp_data_length + (half_warp_data_length - offset_in_warp) - 1;
        } else {
            offset_in_warp = half_warp_data_length - (offset_in_warp - half_warp_data_length) - 1;
        }

        C[warp_data_offset + offset_in_warp] =
            A[warp_data_offset + offset_in_warp] +
            B[warp_data_offset + offset_in_warp];
    }
}

__global__ void vector_add_not_coalesced(int *C, int *A, int *B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        int ii;
        if (i < N / 2) {
            ii = 2 * i;
        } else {
            ii = 2 * (i - (N / 2)) + 1;
        }
        C[ii] = A[ii] + B[ii];
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
    const int threads_per_block = 128;
    int read_offset = 1;
    int write_offset = 1;

    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        read_offset = atoi(argv[2]);
    }
    if (argc > 3) {
        write_offset = atoi(argv[3]);
    }

    assert(N % 2 == 0);

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

    // Transfer the contents of the input host arrays on to the device
    CHECK_CUDA(cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0x00, N * sizeof(int)));

    const int nblocks = (N + threads_per_block - 1) / threads_per_block;

    // warm up driver
    vector_add<<<nblocks, threads_per_block>>>(d_C, d_B, d_A, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    const unsigned long long aligned_start = current_time_ns();
    vector_add<<<nblocks, threads_per_block>>>(d_C, d_B, d_A, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    const unsigned long long aligned_end = current_time_ns();

    const unsigned long long misaligned_read_start = current_time_ns();
    vector_add_read_offset<<<nblocks, threads_per_block>>>(d_C, d_B, d_A, N, read_offset);
    CHECK_CUDA(cudaDeviceSynchronize());
    const unsigned long long misaligned_read_end = current_time_ns();

    const unsigned long long misaligned_write_start = current_time_ns();
    vector_add_write_offset<<<nblocks, threads_per_block>>>(d_C, d_B, d_A, N, write_offset);
    CHECK_CUDA(cudaDeviceSynchronize());
    const unsigned long long misaligned_write_end = current_time_ns();

    const unsigned long long weirdly_coalesced_start = current_time_ns();
    vector_add_weirdly_coalesced<<<nblocks, threads_per_block>>>(d_C, d_B, d_A, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    const unsigned long long weirdly_coalesced_end = current_time_ns();

    const unsigned long long not_coalesced_start = current_time_ns();
    vector_add_not_coalesced<<<nblocks, threads_per_block>>>(d_C, d_B, d_A, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    const unsigned long long not_coalesced_end = current_time_ns();

    printf("Took %llu microseconds on device with aligned accesses\n",
            (aligned_end - aligned_start) / 1000);
    printf("Took %llu microseconds on device with misaligned reads\n",
            (misaligned_read_end - misaligned_read_start) / 1000);
    printf("Took %llu microseconds on device with misaligned writes\n",
            (misaligned_write_end - misaligned_write_start) / 1000);
    printf("Took %llu microseconds on device with weirdly coalesced accesses\n",
            (weirdly_coalesced_end - weirdly_coalesced_start) / 1000);
    printf("Took %llu microseconds on device with uncoalesced accesses\n",
            (not_coalesced_end - not_coalesced_start) / 1000);

    return 0;
}
