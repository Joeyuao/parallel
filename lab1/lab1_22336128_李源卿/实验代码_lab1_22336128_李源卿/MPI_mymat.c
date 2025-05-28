#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

int main(int argc,char **argv){
    
    // printf("%d\n",argc);
    // printf("%s\n",argv[2]);
    //初始化
    MPI_Init(&argc, &argv);
    double *C = NULL;
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    // printf("%d\n",p);
    double start=MPI_Wtime();
    int m , n , k ; 
    double *A = NULL;
    double *B = NULL;
    
    if(rank==0){
        //输入维度
        printf("input the m,n,k:\n");
        scanf("%d%d%d",&m,&n,&k);
        int mnk[3]={m,n,k};
        MPI_Bcast(mnk,3,MPI_INT,0,MPI_COMM_WORLD);
        m=mnk[0];n=mnk[1];k=mnk[2];
        //初始化A和B矩阵
        A=(double*)malloc(m*k*sizeof(double));
        B=(double*)malloc(k*n*sizeof(double));
        C=(double*)malloc(m*n*sizeof(double));
        for (int i = 0; i < m*k; i++)
        {
            A[i]=i+1;
        }
        for (int i = 0; i < k*n; i++)
        {
            B[i]=i+1;
        }
        //规划数据的分组
        int *rows_per_rank = (int *)malloc(p * sizeof(int));
        int *start_row_per_rank = (int *)malloc(p * sizeof(int));
        int remainder = m % p;
        int base_rows = m / p;
        int current_start = 0;
        for (int i = 0; i < p; i++)
        {
            //前remainder组多算一行，刚好不多余
            rows_per_rank[i]=(i<remainder)?base_rows+1:base_rows;
            start_row_per_rank[i]=current_start;
            current_start+=rows_per_rank[i];
        }
        //发送切好片的A矩阵和整个B矩阵，发rows_per_rank[i]是因为接收方需要用来接收矩阵A和发送矩阵C
        for (int i = 1; i < p; i++)
        {
            MPI_Send(&rows_per_rank[i],1,MPI_INT,i,0,MPI_COMM_WORLD);
            MPI_Send(A+start_row_per_rank[i]*k,rows_per_rank[i]*k,MPI_DOUBLE,i,1,MPI_COMM_WORLD);
            MPI_Send(B,k*n,MPI_DOUBLE,i,2,MPI_COMM_WORLD);
        }
        //矩阵乘（调整过循环顺序）
        matrix_multiply(A,B,C,rows_per_rank[0],n,k);
        for (int i = 1; i < p; i++)
        {
            //收集结果
            MPI_Recv(C+start_row_per_rank[i]*n,rows_per_rank[i]*n,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        // print_mat(C,m,n);
        free(A);
        free(B);
        free(C);
        free(rows_per_rank);
        free(start_row_per_rank);
    }
    else{
        //获取矩阵维度
        int mnk[3]={};
        MPI_Bcast(mnk,3,MPI_INT,0,MPI_COMM_WORLD);
        m=mnk[0];n=mnk[1];k=mnk[2];
        //获取当前进程需要算的行数
        int local_rows=0;
        MPI_Recv(&local_rows,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        A=(double*)malloc(local_rows*k*sizeof(double));
        B=(double*)malloc(n*k*sizeof(double));
        double* C=(double*)malloc(local_rows*n*sizeof(double));
        //接受矩阵A，B
        MPI_Recv(A,local_rows*k,MPI_DOUBLE,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(B,n*k,MPI_DOUBLE,0,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        matrix_multiply(A,B,C,local_rows,n,k);
        //返回结果给根进程
        MPI_Send(C,local_rows*n,MPI_DOUBLE,0,0,MPI_COMM_WORLD);
        free(A);
        free(B);
        free(C);
    }
    double time=MPI_Wtime()-start;
    printf("process %d costs %lfs\n",rank,time);
    double max_time;
    MPI_Reduce(&time,&max_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0){
        printf("the max time: %lfs\n",max_time);
    }
    MPI_Finalize();
    return 0;
}