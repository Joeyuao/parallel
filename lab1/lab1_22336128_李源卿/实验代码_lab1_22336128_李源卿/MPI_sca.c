#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
// #include<sys>
// A:m*k B:k*n C:m*n
void matrix_multiply(double *A, double *B, double *C, int rows, int n, int kk) {
    for (int i = 0; i < rows; ++i) {          
        for (int k = 0; k < kk; ++k) {     
            double a = A[i*kk+k];                
            for (int j = 0; j < n; ++j) {  
                C[i*n+j] += a * B[k*n+j];   
            }
        }
    }
}
void print_mat(double* A,int m,int n){
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%lf ",A[i*n+j]);
        }
        printf("\n");
    }
}
int main(int argc, char** argv) {
    
    int rank, size;
    int m, k, n;           // 矩阵维度 A(m×k), B(k×n)
    double *A = NULL;
    double *A_local, *B, *C_local;
    int *rows_per_process = NULL;
    int *displs = NULL;
    int *recvcounts = NULL;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    double start=MPI_Wtime();
    // 主进程读取矩阵维度
    if (rank == 0) {
        printf("Enter matrix dimensions (m k n): ");
        scanf("%d %d %d", &m, &k, &n);
    }
    // 广播维度信息给所有进程
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // 计算每个进程分配的行数（不均匀分配）
    rows_per_process = (int*)malloc(size*sizeof(int));
    displs = (int*)malloc(size*sizeof(int));
    recvcounts = (int*)malloc(size*sizeof(int));
    //规划数据分组
    int base_rows = m / size;
    int remainder = m % size;
    int cur_row = 0;
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            //前remainder个进程多处理一行
            rows_per_process[i]=base_rows+(i<remainder?1:0);
            displs[i]=cur_row*k;  // 每行k个元素
            cur_row+=rows_per_process[i];
            recvcounts[i]=rows_per_process[i]*k; // 每个进程接收的元素数
        }
    }
    // 广播分配策略
    MPI_Bcast(rows_per_process, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(recvcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    // 分配局部内存
    A_local=(double*)malloc(rows_per_process[rank]*k*sizeof(double));
    B=(double*)malloc(k*n*sizeof(double));
    C_local=(double*)malloc(rows_per_process[rank]*n*sizeof(double));
    // 主进程初始化矩阵A和B
    if (rank == 0) {
        A = (double*)malloc(m * k * sizeof(double));
        for (int i = 0; i < m * k; i++) {
            A[i] =i+1;
        }
        for (int i = 0; i < k * n; i++) {
            B[i]=i+1;
        }
    }
    // 分发A的切片和整个B到各进程
    MPI_Scatterv(A, recvcounts, displs, MPI_DOUBLE,
                 A_local, recvcounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    MPI_Bcast(B, k * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // 各进程计算局部C = A_local * B
    matrix_multiply(A_local,B,C_local,rows_per_process[rank],n,k);
    // 准备收集结果的参数（每个进程返回rows_per_process[i]×n个元素）
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            recvcounts[i] = rows_per_process[i] * n;
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
        }
    }
    // 收集结果到主进程
    double* C = NULL;
    if (rank == 0) {
        C = (double*)malloc(m * n * sizeof(double));
    }
    MPI_Gatherv(C_local, rows_per_process[rank] * n, MPI_DOUBLE,
                C, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // 主进程输出结果
    if (rank == 0) {
        // printf("\nMatrix C (%d×%d) = A × B:\n", m, n);
        // print_mat(C,m,n);
        free(A);
        free(C);
    }
    // 释放内存
    free(A_local);
    free(B);
    free(C_local);
    free(rows_per_process);
    free(displs);
    free(recvcounts);
    double time=MPI_Wtime()-start;
    double max_time;
    printf("process %d costs %lfs\n",rank,time);
    MPI_Reduce(&time,&max_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0){
        printf("the max time:%lfs\n",max_time);
    }
    MPI_Finalize();

    return 0;
}