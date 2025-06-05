#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h> 

#define BLOCK_SIZE 16
#define TOLERANCE 1e-5

// 核函数声明
__global__ void share_mulmat_block(float* out, float* A, float* B, int n);
__global__ void share_mulmat_col(float* out, float* A, float* B, int n);
__global__ void share_mulmat_row(float* out, float* A, float* B, int n);
__global__ void ref_matmul(float* out, float* A, float* B, int n);

// 矩阵初始化与验证函数
void initializeMatrix(float* matrix, int n);
bool verifyResults(float* ref, float* test, int n);

double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return -1;
    }
    
    int n = atoi(argv[1]);  // 获取矩阵维度
    size_t size = n * n * sizeof(float);
    
    // 创建主机内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_block = (float*)malloc(size);
    float *h_C_row = (float*)malloc(size);
    float *h_C_col = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);  // 参考结果
    
    // 初始化输入矩阵
    initializeMatrix(h_A, n);
    initializeMatrix(h_B, n);
    
    // 在主机上计算参考结果
    double host_start = getCurrentTime();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += h_A[i * n + k] * h_B[k * n + j];
            }
            h_ref[i * n + j] = sum;
        }
    }
    double host_end = getCurrentTime();
    printf("Host matrix multiplication time: %.3f ms\n", host_end - host_start);
    
    // 创建设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // 复制数据到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 1. 测试分块矩阵乘法
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    double block_start = getCurrentTime();
    share_mulmat_block<<<gridDim, blockDim>>>(d_C, d_A, d_B, n);
    cudaDeviceSynchronize();
    double block_end = getCurrentTime();
    printf("Block matrix multiplication time: %.3f ms\n", block_end - block_start);
    cudaMemcpy(h_C_block, d_C, size, cudaMemcpyDeviceToHost);
    verifyResults(h_ref, h_C_block, n) ? printf("Block result: CORRECT\n") : printf("Block result: INCORRECT\n");
    
    // 2. 测试按行矩阵乘法
    dim3 rowBlock(256);  // 每个块256个线程
    dim3 rowGrid((n + rowBlock.x - 1) / rowBlock.x);
    
    double row_start = getCurrentTime();
    share_mulmat_row<<<rowGrid, rowBlock, n*sizeof(float)>>>(d_C, d_A, d_B, n);
    cudaDeviceSynchronize();
    double row_end = getCurrentTime();
    printf("Row matrix multiplication time: %.3f ms\n", row_end - row_start);
    cudaMemcpy(h_C_row, d_C, size, cudaMemcpyDeviceToHost);
    verifyResults(h_ref, h_C_row, n) ? printf("Row result: CORRECT\n") : printf("Row result: INCORRECT\n");
    
    // 3. 测试按列矩阵乘法
    dim3 colBlock(256);  // 每个块256个线程
    dim3 colGrid((n + colBlock.x - 1) / colBlock.x);
    
    double col_start = getCurrentTime();
    share_mulmat_col<<<colGrid, colBlock, n*sizeof(float)>>>(d_C, d_A, d_B, n);
    cudaDeviceSynchronize();
    double col_end = getCurrentTime();
    printf("Col matrix multiplication time: %.3f ms\n", col_end - col_start);
    cudaMemcpy(h_C_col, d_C, size, cudaMemcpyDeviceToHost);
    verifyResults(h_ref, h_C_col, n) ? printf("Col result: CORRECT\n") : printf("Col result: INCORRECT\n");
    
    // 清理内存
    free(h_A); free(h_B); free(h_ref);
    free(h_C_block); free(h_C_row); free(h_C_col);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}

// 验证函数实现
bool verifyResults(float* ref, float* test, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(ref[i * n + j] - test[i * n + j]) > TOLERANCE) {
                printf("Mismatch at (%d, %d): ref=%.6f, test=%.6f\n", 
                      i, j, ref[i * n + j], test[i * n + j]);
                return false;
            }
        }
    }
    return true;
}

// 初始化矩阵实现
void initializeMatrix(float* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

__global__ void share_mulmat_block(float* out, float* A, float* B, int n) {
    // 静态声明共享内存（大小在编译时确定）
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // 计算当前线程要处理的全局位置
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    float dot = 0.0f;
    
    // 遍历所有分块
    for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
        // 加载A的分块（当前线程加载一个元素）
        int A_col = k * BLOCK_SIZE + tx;
        if (row < n && A_col < n) {
            As[ty][tx] = A[row * n + A_col];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // 加载B的分块（当前线程加载一个元素）
        int B_row = k * BLOCK_SIZE + ty;
        if (B_row < n && col < n) {
            Bs[ty][tx] = B[B_row * n + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads(); // 确保所有线程完成加载
        
        // 计算当前分块的贡献（点积）
        for (int i = 0; i < BLOCK_SIZE; i++) {
            dot += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads(); // 确保所有线程完成计算
    }
    
    // 将结果写回全局内存
    if (row < n && col < n) {
        out[row * n + col] = dot;
    }
}

__global__ void share_mulmat_row(float* out, float* A, float* B, int n) {
    extern __shared__ float B_col[];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    for (int col = 0; col < n; col++) {
        // 协作加载B的一列到共享内存
        for (int ii = 0; ii < n; ii += blockDim.x) {
            int idx = threadIdx.x + ii;
            if (idx < n) {
                B_col[idx] = B[idx * n + col];  // 加载B的第col列
            }
        }
        __syncthreads();  // 确保所有线程完成加载

        // 计算点积
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B_col[k];
        }
        out[row * n + col] = sum;

        __syncthreads();  // 确保所有线程完成计算
    }
}

__global__ void share_mulmat_col(float* out, float* A, float* B, int n) {
    extern __shared__ float A_row[];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;  // 边界检查

    for (int row = 0; row < n; row++) {
        // 协作加载A的一行到共享内存
        for (int ii = 0; ii < n; ii += blockDim.x) {
            int idx = threadIdx.x + ii;
            if (idx < n) {
                A_row[idx] = A[row * n + idx];  // 合并内存访问
            }
        }
        __syncthreads();  // 确保所有线程完成加载

        // 计算点积
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A_row[k] * B[k * n + col];  // 注意B的访问模式
        }
        out[row * n + col] = sum;

        __syncthreads();  // 确保所有线程完成计算
    }
}