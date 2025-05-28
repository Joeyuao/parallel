#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void initialize_matrix(double *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (double)rand() / RAND_MAX; // 随机初始化[0, 1)
    }
}
enum{
    DEFAULT = 0,STATIC,DYNAMIC,GUIDED
}schedule_type;
int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <threads> <size> <schedule>\n", argv[0]);
        printf("Schedule: 0-default, 1-static, 2-dynamic ,3-guided\n");
        return 1;
    }

    int num_threads = atoi(argv[1]);
    int N = atoi(argv[2]);
    int schedule_type = atoi(argv[3]);

    double *A = (double*)malloc(N * N * sizeof(double));
    double *B = (double*)malloc(N * N * sizeof(double));
    double *C = (double*)malloc(N * N * sizeof(double));
    double *local_C = (double*)malloc(N * N * sizeof(double));
    // 初始化矩阵
    initialize_matrix(A, N * N);
    initialize_matrix(B, N * N);

    omp_set_num_threads(num_threads);
    double start = omp_get_wtime();

    // 根据调度类型选择并行策略
    switch (schedule_type) {
        case DEFAULT: // 默认调度
            #pragma omp parallel for
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < N; k++) {
                        sum += A[i*N + k] * B[k*N + j];
                    }
                    C[i*N + j] = sum;
                }
            }
            break;
        case STATIC: // 静态调度
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < N; k++) {
                        sum += A[i*N + k] * B[k*N + j];
                    }
                    C[i*N + j] = sum;
                }
            }
            break;
        case DYNAMIC: // 动态调度
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < N; k++) {
                        sum += A[i*N + k] * B[k*N + j];
                    }
                    C[i*N + j] = sum;
                }
            }
            break;
        case GUIDED: // 动态调度
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < N; k++) {
                    sum += A[i*N + k] * B[k*N + j];
                }
                C[i*N + j] = sum;
            }
        }
            break;
        default:
            printf("Invalid schedule type\n");
            return 1;
    }

    double end = omp_get_wtime();
    printf("Execution time: %.4f seconds\n", end - start);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            local_C[i*N + j] = sum;
        }
    }
    int iserr=0;
    for (int i = 0; i < N*N; i++)
    {
        if (local_C[i] == C[i])
        {
            continue;
        }
        else
        {
            iserr = 1;
            printf("wrong!\n");
            break;
        }
    }
    if(!iserr){
        printf("right!\n");
    }
    free(A);
    free(B);
    free(C);
    free(local_C);
    return 0;
}