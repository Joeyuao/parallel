#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
int BLOCK_SIZE;
#define MAX_THREADS 16

// 声明全局共享矩阵
double *A, *B;
int start_row;
int end_row=0;
int N;
typedef struct {
    int thread_id;
    
    double *C;  // 结果矩阵C仍通过参数传递
} ThreadArg;
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
void *matrix_mult(void *arg) {
    ThreadArg *t_arg = (ThreadArg *)arg;
    while (1)
    {
        int st,ed;
        pthread_mutex_lock(&mtx);
        start_row += BLOCK_SIZE;
        end_row += BLOCK_SIZE;
        if (start_row >= N)
        {
            pthread_mutex_unlock(&mtx);
            pthread_exit(NULL);
        }
        
        end_row = end_row < N ? end_row : N;
        st = start_row; ed = end_row;
        // printf("thread:  %ld, start: %d  , end:  %d\n", pthread_self()%100, start_row, end_row);
        pthread_mutex_unlock(&mtx);
        for (int i = st; i < ed; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    // 直接访问全局变量A和B
                    sum += A[i * N + k] * B[k * N + j];
                }
                t_arg->C[i * N + j] = sum;
            }
        }
    
    }
    pthread_exit(NULL);
    
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <matrix size> <number of threads> <BLOCK_PERCENT>\n", argv[0]);
        return 1;
    }

    N = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    if ( num_threads < 1 || num_threads > MAX_THREADS) {
        fprintf(stderr, "Invalid input parameters.\n");
        return 1;
    }
    int BLOCK_PERCENT = atoi(argv[3]);
    BLOCK_SIZE = N / num_threads / BLOCK_PERCENT;
    if(BLOCK_SIZE == 0){
        printf("please check your input!\n");
        return 2;
    }
        start_row = -BLOCK_SIZE;
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

    // // 串行计算
    // struct timeval serial_start, serial_end;
    // gettimeofday(&serial_start, NULL);
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         double sum = 0.0;
    //         for (int k = 0; k < N; k++) {
    //             sum += A[i * N + k] * B[k * N + j];
    //         }
    //         C_serial[i * N + j] = sum;
    //     }
    // }
    // gettimeofday(&serial_end, NULL);
    // double serial_time = (serial_end.tv_sec - serial_start.tv_sec) +
    //                        (serial_end.tv_usec - serial_start.tv_usec) / 1e6;
    // 并行计算
    pthread_t threads[num_threads];
    ThreadArg args[num_threads];
    int rows_per_thread = N / num_threads;
    int remaining = N % num_threads;
    int current_start = 0;

    struct timeval start_parallel, end_parallel;
    gettimeofday(&start_parallel, NULL);

    for (int i = 0; i < num_threads; i++) {
        args[i].thread_id = i;
        
        args[i].C = C_parallel;
        pthread_create(&threads[i], NULL, matrix_mult, &args[i]);

    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    gettimeofday(&end_parallel, NULL);
    double parallel_time = (end_parallel.tv_sec - start_parallel.tv_sec) +
                           (end_parallel.tv_usec - start_parallel.tv_usec) / 1e6;

    // 验证结果
    // int correct = 1;
    // for (int i = 0; i < N * N; i++) {
    //     if (fabs(C_serial[i] - C_parallel[i]) != 0) {
    //         correct = 0;
    //         break;
    //     }
    // }

    // 输出结果
    // printf("Matrix Size: %d, Threads: %d\n", N, num_threads);
    // printf("Serial Time: %.8f s\n", serial_time);
    printf("Parallel Time: %.8f s\n", parallel_time);
    // printf("Speedup: %.2f\n", serial_time / parallel_time);
    // printf("Efficiency: %.2f%%\n", (serial_time / parallel_time) / num_threads * 100);
    // printf("Result: %s\n\n", correct ? "Correct" : "Incorrect");
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