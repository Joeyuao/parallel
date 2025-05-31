#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "parallel_for.h"
#include <sys/time.h>
struct FuncArgs {
    int size;
    double *A;
    double *B;
    double *C;
};

void initialize_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (double)rand() / RAND_MAX;
    }
}

void* func(int cur, void *local_args) {
    struct FuncArgs *args = (struct FuncArgs*)local_args;
    for (int j = 0; j < args->size; j++) {
        double sum = 0.0;
        for (int k = 0; k < args->size; k++) {
            sum += args->A[cur * args->size + k] * args->B[k * args->size + j];
        }
        args->C[cur * args->size + j] = sum;
    }
    return NULL;
}

enum { DEFAULT = 0, STATIC, DYNAMIC, GUIDED };

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <threads> <size> <schedule>\n", argv[0]);
        printf("Schedule: 0-default, 1-static, 2-dynamic, 3-guided\n");
        return 1;
    }

    int num_threads = atoi(argv[1]);
    int N = atoi(argv[2]);
    int schedule = atoi(argv[3]);

    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));
    double *local_C = malloc(N * N * sizeof(double));

    // 初始化矩阵
    initialize_matrix(A, N*N);
    initialize_matrix(B, N*N);

    struct FuncArgs args = {
        .size = N,
        .A = A,
        .B = B,
        .C = C
    };
    struct timeval st, ed;
    gettimeofday(&st, NULL);

    // 调用 parallel_for 实现并行计算
    switch (schedule) {
        case DEFAULT:
        case STATIC:  // 默认和静态调度均使用相同的并行策略
            parallel_for(0, N, 1, func, &args, num_threads);
            break;
        case DYNAMIC:
        case GUIDED:
            printf("Warning: DYNAMIC/GUIDED not supported, fallback to STATIC\n");
            parallel_for(0, N, 1, func, &args, num_threads);
            break;
        default:
            printf("Invalid schedule type\n");
            return 1;
    }

    gettimeofday(&ed, NULL);
    printf("Execution time: %.4f seconds\n", ed.tv_sec - st.tv_sec+(ed.tv_usec - st.tv_usec) * 1e-6);

    // 串行计算验证结果
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            local_C[i*N + j] = sum;
        }
    }

    // 结果校验
    int iserr = 0;
    for (int i = 0; i < N*N; i++) {
        if (C[i] != local_C[i]) {
            iserr = 1;
            printf("Mismatch at index %d: %.6f vs %.6f\n", i, C[i], local_C[i]);
            break;
        }
    }
    printf(iserr ? "Result: Wrong\n" : "Result: Correct\n");
    printf("%d",iserr);
    free(A);
    free(B);
    free(C);
    free(local_C);
    return 0;
}