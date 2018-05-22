#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"

#define CHECK_CUDA(call) { \
    const cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA ERROR @ %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

/*
 * A simple example of performing matrix-vector multiplication using the cuBLAS
 * library and some randomly generated inputs.
 */

/*
 * M = # of rows
 * N = # of columns
 */
int M = 1024;
int N = 1024;

/*
 * Generate a vector of length N with random single-precision floating-point
 * values between 0 and 100.
 */
void generate_random_vector(int N, float **outX)
{
    int i;
    double rMax = (double)RAND_MAX;
    float *X = (float *)malloc(sizeof(float) * N);

    for (i = 0; i < N; i++)
    {
        int r = rand();
        double dr = (double)r;
        X[i] = (dr / rMax) * 10.0;
    }

    *outX = X;
}

/*
 * Verify that Y = M * X
 */
static void verify(float *A, float *X, float *Y, int M, int N, float alpha) {
    double avg_perc_err = 0.0;
    for (int row = 0; row < M; row++) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += alpha * A[row * N + col] * X[col];
        }

        avg_perc_err += fabs(Y[row] - sum) / sum;
    }
    avg_perc_err /= (float)M;
    printf("\n%% error = %f%%\n", 100.0 * avg_perc_err);
}

/*
 * Generate a matrix with M rows and N columns in column-major order. The matrix
 * will be filled with random single-precision floating-point values between 0
 * and 100.
 */
void generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);

    // For each column
    for (j = 0; j < N; j++)
    {
        // For each row
        for (i = 0; i < M; i++)
        {
            double dr = (double)rand();
            A[j * M + i] = (dr / rMax) * 100.0;
        }
    }

    *outA = A;
}

int main(int argc, char **argv)
{
    int i;
    float *A, *dA;
    float *X, *dX;
    float *Y, *dY;
    float beta;
    float alpha;
    cublasHandle_t handle = 0;

    alpha = 3.0f;
    beta = 4.0f;

    // Generate inputs
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_vector(N, &X);
    Y = (float *)malloc(sizeof(float) * M);
    memset(Y, 0x00, sizeof(float) * M);

    // Create the cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&handle));

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK_CUDA(cudaMalloc((void **)&dX, sizeof(float) * N));
    CHECK_CUDA(cudaMalloc((void **)&dY, sizeof(float) * M));

    // Transfer inputs to the device
    CHECK_CUBLAS(cublasSetVector(N, sizeof(float), X, 1, dX, 1));
    CHECK_CUBLAS(cublasSetVector(M, sizeof(float), Y, 1, dY, 1));
    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(float), A, M, dA, M));

    // Execute the matrix-vector multiplication
    CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, dA, M, dX, 1,
                             &beta, dY, 1));

    // Retrieve the output vector from the device
    CHECK_CUBLAS(cublasGetVector(M, sizeof(float), dY, 1, Y, 1));

    for (i = 0; i < 10; i++)
    {
        printf("%2.2f\n", Y[i]);
    }

    printf("...\n");

    verify(A, X, Y, M, N, alpha);

    free(A);
    free(X);
    free(Y);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dY));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
