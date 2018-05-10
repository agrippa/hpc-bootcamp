#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"

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
        X[i] = (dr / rMax) * 100.0;
    }

    *outX = X;
}

/*
 * Verify that Y = M * X
 */
static void verify(float *A, float *X, float *Y, int M, int N) {
    for (int row = 0; row < M; row++) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += A[row * N + col] * X[col];
        }

        if (sum != Y[row]) {
            fprintf(stderr, "Value mismatch at Y[%d].\n", row);
            fprintf(stderr, "Expected = %f\n", sum);
            fprintf(stderr, "CUBLAS   = %f\n", Y[row]);
            exit(1);
        }
    }
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

    alpha = 3.0f;
    beta = 4.0f;

    // Generate inputs
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_vector(N, &X);
    Y = (float *)malloc(sizeof(float) * M);

    // TODO 1. Declare and create a CUBLAS handle using cublasCreate

    /*
     * TODO 2. Allocate a float array on the device with M x N elements, to
     * store the matrix 'A' (cudaMalloc).
     */

    /*
     * TODO 3. Allocate a float array on the device with N elements, to
     * store the input vector 'X' (cudaMalloc).
     */

    /*
     * TODO 4. Allocate a float array on the device with M elements, to
     * store the output vector 'Y' (cudaMalloc).
     */

    /*
     * TODO 5. Copy the input M x N matrix 'A' to the space you've allocated for
     * it on the device, using cublasSetMatrix.
     */

    /*
     * TODO 6. Copy the input N-element vector 'X' to the space you've allocated
     * for it on the device, using cublasSetVector.
     */

    /*
     * TODO 7. Call cublasSgemv to perform the dense matrix-vector multiplication
     * M*X=Y. You may assume:
     *
     *   1. We do not wish to do a transpose.
     *   2. alpha = 3.0f (defined above).
     *   3. beta = 4.0f (defined above).
     */

    /*
     * TODO 8. Copy the output M-element vector 'Y' out of the CUDA device and
     * into the host buffer 'Y', using cublasGetVector.
     */

    for (i = 0; i < 10; i++)
    {
        printf("%2.2f\n", Y[i]);
    }

    printf("...\n");

    verify(A, X, Y, M, N);

    free(A);
    free(X);
    free(Y);

    /*
     * TODO 9. Free any device arrays you have allocated, using cudaFree.
     * Release the CUBLAS handle created, using cublasDestroy.
     */

    return 0;
}
