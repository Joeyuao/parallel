#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#define MAX_THREADS 16

// 声明全局共享矩阵
double *A, *B;

typedef struct {
    int start_row;
    int end_row;
    int N;
    double *C;
} __attribute__((aligned(64)))
void *matrix_mult(void *arg) {
    ThreadArg *t_arg = (ThreadArg *)arg;
    for (int i = t_arg->start_row; i < t_arg->end_row; i++) {
        for (int j = 0; j < t_arg->N; j++) {
            double sum = 0.0;
            for (int k = 0; k < t_arg->N; k++) {
                // 直接访问全局变量A和B
                sum += A[i * t_arg->N + k] * B[k * t_arg->N + j];
            }
            t_arg->C[i * t_arg->N + j] = sum;
        }
    }
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <matrix size> <number of threads>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    if ( num_threads < 1 || num_threads > MAX_THREADS) {
        fprintf(stderr, "Invalid input parameters.\n");
        return 1;
    }

    // 分配全局矩阵内存
    A = malloc(N * N * sizeof(double));
    B = malloc(N * N * sizeof(double));
    double *C_serial = malloc(N * N * sizeof(double));
    double *C_parallel = malloc(N * N * sizeof(double));

    // 初始化矩阵
    srand(time(NULL));
    for (int i = 0; i < N * N; i++) {
        // A[i] = (double)rand() / RAND_MAX;
        // B[i] = (double)rand() / RAND_MAX;
        A[i] = i+1;
        B[i] = i+1;
    }

    // 串行计算
    struct timeval serial_start, serial_end;
    gettimeofday(&serial_start, NULL);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C_serial[i * N + j] = sum;
        }
    }
    gettimeofday(&serial_end, NULL);
    double serial_time = (serial_end.tv_sec - serial_start.tv_sec) +
                           (serial_end.tv_usec - serial_start.tv_usec) / 1e6;

    // 并行计算
    pthread_t threads[num_threads];
    ThreadArg args[num_threads];
    int rows_per_thread = N / num_threads;
    int remaining = N % num_threads;
    int current_start = 0;

    struct timeval start_parallel, end_parallel;
    gettimeofday(&start_parallel, NULL);

    for (int i = 0; i < num_threads; i++) {
        args[i].start_row = current_start;
        args[i].end_row = current_start + rows_per_thread + (i < remaining ? 1 : 0);
        args[i].N = N;
        args[i].C = C_parallel;
        pthread_create(&threads[i], NULL, matrix_mult, &args[i]);
        current_start = args[i].end_row;
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    gettimeofday(&end_parallel, NULL);
    double parallel_time = (end_parallel.tv_sec - start_parallel.tv_sec) +
                           (end_parallel.tv_usec - start_parallel.tv_usec) / 1e6;

    // 验证结果
    int correct = 1;
    for (int i = 0; i < N * N; i++) {
        if (fabs(C_serial[i] - C_parallel[i]) != 0) {
            correct = 0;
            break;
        }
    }

    // 输出结果
    printf("Matrix Size: %d, Threads: %d\n", N, num_threads);
    printf("Serial Time: %.8f s\n", serial_time);
    printf("Parallel Time: %.8f s\n", parallel_time);
    printf("Speedup: %.2f\n", serial_time / parallel_time);
    printf("Efficiency: %.2f%%\n", (serial_time / parallel_time) / num_threads * 100);
    printf("Result: %s\n\n", correct ? "Correct" : "Incorrect");
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         printf("%lf ",C_parallel[i*N+j]);
    //     }
    //     printf("\n");
    // }
    
    // 释放内存
    free(A);
    free(B);
    free(C_serial);
    free(C_parallel);
    return 0;
}