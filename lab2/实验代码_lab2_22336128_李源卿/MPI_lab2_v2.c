#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define MAT_SIZE 128
// 矩阵乘法函数，用于计算局部矩阵乘法
void matrix_multiply(double *A, double *B, double *C, int rows, int n, int kk) {
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < kk; ++k) {
            double a = A[i * kk + k];
            for (int j = 0; j < n; ++j) {
                C[i * n + j] += a * B[k * n + j];
            }
        }
    }
}

// 打印矩阵函数，用于输出矩阵元素
void print_mat(double *A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%lf ", A[i * n + j]);
        }
        printf("\n");
    }
}

// 定义结构体 mkn，用于存储矩阵的维度信息
typedef struct {
    int m;
    int k;
    int n;
} mkn;

int main(int argc, char **argv) {
    int rank, size;
    double *A = NULL;
    double *A_local, *B, *C_local;
    int *rows_per_process = NULL;
    int *displs = NULL;
    int *recvcounts = NULL;
    mkn MKN;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 主进程（rank 为 0）读取矩阵维度信息
    // if (rank == 0) {
    //     printf("Enter matrix dimensions (m k n): ");
    //     scanf("%d %d %d", &MKN.m, &MKN.k, &MKN.n);
    // }
    MKN.k=MKN.m=MKN.n=MAT_SIZE;
    // 记录开始时间
    double start = MPI_Wtime();

    // 创建自定义数据类型 mkn_type，用于广播矩阵维度信息
    MPI_Datatype mkn_type;
    {
        int blocklengths[3] = {1, 1, 1};
        MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_INT};
        MPI_Aint displacements[3];
        displacements[0] = 0;
        displacements[1] = 4;
        displacements[2] = 8;
        // 创建自定义数据类型
        MPI_Type_create_struct(3, blocklengths, displacements, types, &mkn_type);
        // 提交自定义数据类型
        MPI_Type_commit(&mkn_type);
    }

    // 广播矩阵维度信息给所有进程
    MPI_Bcast(&MKN, 1, mkn_type, 0, MPI_COMM_WORLD);
    // 释放自定义数据类型
    MPI_Type_free(&mkn_type);

    // 为每个进程分配的行数、偏移量和接收元素数数组分配内存
    rows_per_process = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));
    recvcounts = (int *)malloc(size * sizeof(int));

    // 计算每个进程分配的行数（不均匀分配）
    int base_rows = MKN.m / size;
    int remainder = MKN.m % size;
    int cur_row = 0;
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            // 前 remainder 个进程多处理一行
            rows_per_process[i] = base_rows + (i < remainder ? 1 : 0);
            displs[i] = cur_row * MKN.k;
            cur_row += rows_per_process[i];
            recvcounts[i] = rows_per_process[i] * MKN.k;
        }
    }

    MPI_Datatype size_len_block;
    {
        int blocklengths = size;
        MPI_Datatype types = MPI_INT;
        MPI_Aint displacements = 0;
        // 创建自定义数据类型
        MPI_Type_create_struct(1, &blocklengths, &displacements, &types, &size_len_block);
        // 提交自定义数据类型
        MPI_Type_commit(&size_len_block);
    }
    // 广播分配策略给所有进程
    MPI_Bcast(rows_per_process, 1, size_len_block, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, 1, size_len_block, 0, MPI_COMM_WORLD);
    MPI_Bcast(recvcounts, 1, size_len_block, 0, MPI_COMM_WORLD);

    // 为每个进程分配局部内存
    A_local = (double *)malloc(rows_per_process[rank] * MKN.k * sizeof(double));
    B = (double *)malloc(MKN.k * MKN.n * sizeof(double));
    C_local = (double *)malloc(rows_per_process[rank] * MKN.n * sizeof(double));

    // 主进程初始化矩阵 A 和 B
    if (rank == 0) {
        A = (double *)malloc(MKN.m * MKN.k * sizeof(double));
        for (int i = 0; i < MKN.m * MKN.k; i++) {
            A[i] = i + 1;
        }
        for (int i = 0; i < MKN.k * MKN.n; i++) {
            B[i] = i + 1;
        }
    }

    // 主进程将矩阵 A 切片分发给各进程
    MPI_Scatterv(A, recvcounts, displs, MPI_DOUBLE, A_local, recvcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 创建自定义数据类型表示 B 矩阵
    MPI_Datatype b_matrix_type;
    {
        int blocklengths = MKN.k * MKN.n;
        MPI_Datatype types = MPI_DOUBLE;
        MPI_Aint displacements = 0;
        // 创建自定义数据类型
        MPI_Type_create_struct(1, &blocklengths, &displacements, &types, &b_matrix_type);
        // 提交自定义数据类型
        MPI_Type_commit(&b_matrix_type);
    }

    // 使用自定义数据类型广播 B 矩阵
    MPI_Bcast(B, 1, b_matrix_type, 0, MPI_COMM_WORLD);

    // 释放自定义数据类型
    MPI_Type_free(&b_matrix_type);

    // 各进程计算局部矩阵乘法结果
    matrix_multiply(A_local, B, C_local, rows_per_process[rank], MKN.n, MKN.k);

    // 准备收集结果的参数
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            recvcounts[i] = rows_per_process[i] * MKN.n;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
        }
    }

    // 各进程将局部结果收集到主进程
    double *C = NULL;
    if (rank == 0) {
        C = (double *)malloc(MKN.m * MKN.n * sizeof(double));
    }
    MPI_Gatherv(C_local, rows_per_process[rank] * MKN.n, MPI_DOUBLE, C, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 主进程输出结果并释放相关内存
    if (rank == 0) {
        printf("Print mat:\n");
        print_mat(C, MKN.m, MKN.n);
        free(A);
        free(C);
    }

    // 释放各进程的局部内存
    free(A_local);
    free(B);
    free(C_local);
    free(rows_per_process);
    free(displs);
    free(recvcounts);

    // 计算本进程的执行时间
    double time = MPI_Wtime() - start;
    double max_time;
    printf("process %d costs %lfs\n", rank, time);

    // 归约计算所有进程中的最大执行时间
    MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("the max time:%lfs\n", max_time);
        
        // print_mat(C,MKN.m,MKN.n);
    }
    
    // 结束 MPI 环境
    MPI_Finalize();
    return 0;
}