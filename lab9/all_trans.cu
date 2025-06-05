#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h> 
#define BDIM 32
__global__ void trans(int* out, int* in, int n) {
    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by + ty;
    int col = bx + tx;
    
    if (row < n && col < n) {
        out[col * n + row] = in[row * n + col];
    }
}

__global__ void trans_conflict(int* out, int* in, int n) {
    __shared__ int smem[BDIM * BDIM];

    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by + ty;
    int col = bx + tx;
    
    if (row < n && col < n) {
        smem[ty*BDIM + tx] = in[row * n + col];
    }
    __syncthreads();
    
    if (row < n && col < n) {
        out[(bx+ty)*n + by+tx] = smem[tx*BDIM + ty];
    }
}

__global__ void trans_solve_conflict0(int* out, int* in, int n) {
    __shared__ int smem[BDIM * BDIM];

    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by + ty;
    int col = bx + tx;
    
    if (row < n && col < n) {
        smem[ty*BDIM + tx] = in[row * n + col];
    }
    __syncthreads();
    
    if (row < n && col < n) {
        out[col * n + row] = smem[ty*BDIM + tx];
    }
}

__global__ void trans_solve_conflict1(int* out, int* in, int n) {
    __shared__ int smem[BDIM * (BDIM+1)];

    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by + ty;
    int col = bx + tx;
    
    if (row < n && col < n) {
        smem[ty*(BDIM+1) + tx] = in[row * n + col];
    }
    __syncthreads();
    
    if (row < n && col < n) {
        out[(bx+ty)*n + by+tx] = smem[tx*(BDIM+1) + ty];
    }
}
void initializeMatrix(int* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i*n + j] = i * n + j;  
        }
    }
}

void printMatrix(int* matrix, int n, int size) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%3d ", matrix[i*size + j]);
        }
        printf("\n");
    }
}
int frac(int n_2){
    double sq = sqrt(double(n_2));
    for (int i = sq; i >= 1; i--){
        if(n_2 % i == 0){
            return i;
        }
    }
    return -1;
}
double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
int main(int argc,char**argv) {
    if (argc != 3) {
        printf("Usage: %s <matrix_size> <block_size>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int B_size = atoi(argv[2]);
    size_t size = n * n * sizeof(int);
    
    // 记录程序总开始时间
    double total_start = getCurrentTime();
    
    // 分配和初始化主机内存
    int* h_in = (int*)malloc(size);
    int* h_out = (int*)malloc(size);
    initializeMatrix(h_in, n);
    
    // 分配设备内存
    int *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    
    // 拷贝数据到设备
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    // 计算网格和块维度
    int G_size2 = (n + B_size*B_size-1) / (B_size*B_size);
    int G_size_x = frac(G_size2);
    int G_size_y = G_size2 / G_size_x;
    dim3 BlockDim(B_size, B_size);
    dim3 GridDim(G_size_x, G_size_y);
    
    // 创建CUDA事件用于精确计时
    cudaEvent_t start, stop;
    float avg = 0.0;
    float cnt = 111.0;
    for (float i = 0; i <= cnt; i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // 记录内核开始时间
        cudaEventRecord(start);
        
        // 启动内核
        trans_solve_conflict1<<<GridDim, BlockDim>>>(d_out, d_in, n);
        
        // 记录内核结束时间
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        // 计算内核执行时间
        float kernel_time = 0;
        cudaEventElapsedTime(&kernel_time, start, stop);
        avg += kernel_time;
    }
    avg = avg / cnt;    
    
    // 拷贝结果回主机
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    // 记录程序总结束时间
    double total_end = getCurrentTime();
    
    // 打印结果验证
    printf("Original matrix (top-left 5x5):\n");
    printMatrix(h_in, 5, n);
    printf("\nTransposed matrix (top-left 5x5):\n");
    printMatrix(h_out, 5, n);
    
    // 打印计时结果
    printf("\nPerformance Metrics:\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Block size: %d x %d\n", B_size, B_size);
    printf("Grid size: %d x %d\n", G_size_x, G_size_y);
    printf("Kernel execution time: %.6f ms\n", avg);
    printf("Total program time: %.3f ms\n", total_end - total_start);
    
    // 清理
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaDeviceSynchronize();
    return 0;
}