#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void matrix_multiply(double *A, double *B, double *C, int rows, int n, int k) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0;
            for (int l = 0; l < n; l++) {
                C[i * k + j] += A[i * n + l] * B[l * k + j];
            }
        }
    }
}

int main(int argc ,char **argv) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    int m = 4, n = 4, k = 4; // 示例矩阵维度，可根据需要修改
    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    if (rank == 0) {
        // 初始化矩阵A和B
        A = (double *)malloc(m * n * sizeof(double));
        B = (double *)malloc(n * k * sizeof(double));
        C = (double *)malloc(m * k * sizeof(double));

        // 填充示例数据
        for (int i = 0; i < m * n; i++) A[i] = i + 1;
        for (int i = 0; i < n * k; i++) B[i] = i + 1;

        // 计算每个进程的行数及起始行
        int *rows_per_rank = (int *)malloc(p * sizeof(int));
        int *start_row_per_rank = (int *)malloc(p * sizeof(int));
        int remainder = m % p;
        int base_rows = m / p;
        int current_start = 0;

        for (int i = 0; i < p; i++) {
            rows_per_rank[i] = (i < remainder) ? base_rows + 1 : base_rows;
            start_row_per_rank[i] = current_start;
            current_start += rows_per_rank[i];
        }

        // 分发数据到各进程
        for (int i = 1; i < p; i++) {
            int rows_i = rows_per_rank[i];
            if (rows_i == 0) continue;

            int meta[3] = {rows_i, n, k};
            MPI_Send(meta, 3, MPI_INT, i, 0, MPI_COMM_WORLD);

            double *A_block = A + start_row_per_rank[i] * n;
            MPI_Send(A_block, rows_i * n, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);

            MPI_Send(B, n * k, MPI_DOUBLE, i, 2, MPI_COMM_WORLD);
        }

        // 处理主进程自身的计算部分
        int rows_0 = rows_per_rank[0];
        if (rows_0 > 0) {
            double *A_block = A + start_row_per_rank[0] * n;
            double *C_block = C + start_row_per_rank[0] * k;
            matrix_multiply(A_block, B, C_block, rows_0, n, k);
        }

        // 接收并整合其他进程的结果
        for (int i = 1; i < p; i++) {
            int rows_i = rows_per_rank[i];
            if (rows_i == 0) continue;

            double *buffer = (double *)malloc(rows_i * k * sizeof(double));
            MPI_Recv(buffer, rows_i * k, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            int start_row = start_row_per_rank[i];
            for (int r = 0; r < rows_i; r++) {
                memcpy(C + (start_row + r) * k, buffer + r * k, k * sizeof(double));
            }
            free(buffer);
        }

        // 输出结果（示例）
        printf("Matrix C:\n");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                printf("%8.2f", C[i * k + j]);
            }
            printf("\n");
        }

        free(A);
        free(B);
        free(C);
        free(rows_per_rank);
        free(start_row_per_rank);
    } else {
        // 从进程代码
        int meta[3];
        MPI_Recv(meta, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int rows_i = meta[0];
        n = meta[1];
        k = meta[2];

        if (rows_i == 0) {
            MPI_Finalize();
            return 0;
        }

        double *A_block = (double *)malloc(rows_i * n * sizeof(double));
        double *B = (double *)malloc(n * k * sizeof(double));
        double *C_block = (double *)malloc(rows_i * k * sizeof(double));

        MPI_Recv(A_block, rows_i * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(B, n * k, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        matrix_multiply(A_block, B, C_block, rows_i, n, k);

        MPI_Send(C_block, rows_i * k, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);

        free(A_block);
        free(B);
        free(C_block);
    }

    MPI_Finalize();
    return 0;
}