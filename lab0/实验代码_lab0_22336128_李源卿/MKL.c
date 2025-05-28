#define min(x,y) (((x) < (y)) ? (x) : (y))
#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include <time.h>
#include <sys/time.h> // 添加高精度计时头文件

int main()
{
    double *A, *B, *C;
    int m, n, k, i, j;
    double alpha, beta;
    struct timeval start, end; // 使用gettimeofday

    printf("\n This example computes real matrix C=alpha*A*B+beta*C using \n"
           " Intel(R) MKL function dgemm, where A, B, and C are matrices and \n"
           " alpha and beta are double precision scalars\n\n");

    m = 1000, k = 1000, n = 1000;
    printf(" Initializing data for matrix multiplication C=A*B for matrix \n"
           " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
    alpha = 1.0; beta = 0.0;

    printf(" Allocating memory for matrices aligned on 64-byte boundary for better \n"
           " performance \n\n");
    A = (double *)mkl_malloc(m * k * sizeof(double), 64);
    B = (double *)mkl_malloc(k * n * sizeof(double), 64);
    C = (double *)mkl_malloc(m * n * sizeof(double), 64);
    if (A == NULL || B == NULL || C == NULL) {
        printf("\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }

    printf(" Intializing matrix data \n\n");
    for (i = 0; i < (m * k); i++) {
        A[i] = (double)rand() / RAND_MAX;
    }

    for (i = 0; i < (k * n); i++) {
        B[i] = (double)rand() / RAND_MAX;
    }

    for (i = 0; i < (m * n); i++) {
        C[i] = 0.0;
    }

    printf(" Computing matrix product using Intel(R) MKL dgemm function via CBLAS interface \n\n");
    gettimeofday(&start, NULL); // 开始计时
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    gettimeofday(&end, NULL); // 结束计时

    // 计算时间差（微秒级精度）
    double time_used = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("time cost: %.6f seconds\n", time_used);

    printf("\n Computations completed.\n\n");

    // printf(" Top left corner of matrix A: \n");
    // for (i = 0; i < min(m, 6); i++) {
    //     for (j = 0; j < min(k, 6); j++) {
    //         printf("%12.0f", A[j + i * k]);
    //     }
    //     printf("\n");
    // }

    // printf("\n Top left corner of matrix B: \n");
    // for (i = 0; i < min(k, 6); i++) {
    //     for (j = 0; j < min(n, 6); j++) {
    //         printf("%12.0f", B[j + i * n]);
    //     }
    //     printf("\n");
    // }

    // printf("\n Top left corner of matrix C: \n");
    // for (i = 0; i < min(m, 6); i++) {
    //     for (j = 0; j < min(n, 6); j++) {
    //         printf("%12.5G", C[j + i * n]);
    //     }
    //     printf("\n");
    // }

    printf("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    printf(" Example completed. \n\n");
    return 0;
}