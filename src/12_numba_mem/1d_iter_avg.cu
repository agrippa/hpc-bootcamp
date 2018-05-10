#include <stdio.h>
#include <sys/time.h>

#define CHECK_CUDA(func) { \
    const cudaError_t err = (func); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error @ %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

static double seconds() {
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


__global__ void kernel(float *nxt, float *curr, const int N) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        tid++;
        nxt[tid] = (curr[tid - 1] + curr[tid + 1]) / 2.0;
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <N> <niters>\n", argv[0]);
        return 1;
    }
    const int N = atoi(argv[1]);
    const int niters = atoi(argv[2]);

    float *seed = (float *)malloc((N + 2) * sizeof(*seed));
    memset(seed, 0x00, (N + 2) * sizeof(*seed));
    seed[N + 1] = 1.0;

    float *d_curr, *d_nxt;
    CHECK_CUDA(cudaMalloc((void **)&d_curr, (N + 2) * sizeof(*d_curr)));
    CHECK_CUDA(cudaMalloc((void **)&d_nxt, (N + 2) * sizeof(*d_nxt)));
    CHECK_CUDA(cudaMemcpy(d_curr, seed, (N + 2) * sizeof(*d_curr),
                cudaMemcpyHostToDevice));

    double start_time = seconds();
    for (int iter = 0; iter < niters; iter++) {
        const int threads_per_block = 256;
        const int blocks_per_grid = (N + threads_per_block - 1) /
            threads_per_block;
        kernel<<<blocks_per_grid, threads_per_block>>>(d_nxt, d_curr, N);
        float *tmp = d_nxt;
        d_nxt = d_curr;
        d_curr = tmp;
    }
    CHECK_CUDA(cudaMemcpy(seed, d_curr,  (N + 2) * sizeof(*seed),
                cudaMemcpyDeviceToHost));
    double elapsed_time = seconds() - start_time;
    printf("Elapsed time for N=%d, # iters=%d is %f s\n", N, niters, elapsed_time);
    printf("%f iters / s\n", (float)niters / elapsed_time);

    return 0;
}
