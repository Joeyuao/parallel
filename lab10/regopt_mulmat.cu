#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

// 常量定义
#define TOLERANCE 1e-5
#define RC_TILE_SIZE 16
#define BLOCK_SIZE 16
#define TILE_SIZE 4

// 核函数声明
__global__ void row_optimized_matmul(float* out, const float* A, const float* B, int n);
__global__ void col_optimized_matmul(float* out, const float* A, const float* B, int n);
__global__ void block_optimized_matmul(float* out, const float* A, const float* B, int n);
__global__ void ref_matmul(float* out, float* A, float* B, int n);

// 辅助函数声明
void initializeMatrix(float* matrix, int n);
bool verifyResults(float* ref, float* test, int n);
double getCurrentTime();

int main(int argc, char** argv) {
    // if (argc != 2) {
    //     printf("Usage: %s <matrix_size>\n", argv[0]);
    //     return -1;
    // }
    
    // int n = atoi(argv[1]);
    int n = 512;
    size_t size = n * n * sizeof(float);
    
    // 检查矩阵大小是否兼容
    // if (n % RC_TILE_SIZE != 0 || n % (BLOCK_SIZE * TILE_SIZE) != 0) {
    //     printf("Error: Matrix size must be divisible by %d and %d\n", 
    //            RC_TILE_SIZE, BLOCK_SIZE * TILE_SIZE);
    //     return 1;
    // }

    // 创建主机内存（保持不变）
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_row = (float*)malloc(size);
    float *h_C_col = (float*)malloc(size);
    float *h_C_block = (float*)malloc(size);
    float *h_ref = (float*)malloc(size);
    
    // 初始化矩阵（保持不变）
    initializeMatrix(h_A, n);
    initializeMatrix(h_B, n);
    
    // 计算参考结果（保持不变）
    double host_start = getCurrentTime();
    // ref_matmul<<<1, 1>>>(h_ref, h_A, h_B, n);
    // cudaDeviceSynchronize();
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
    printf("Host reference computation time: %.3f ms\n", host_end - host_start);
    
    // 创建设备内存（保持不变）
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // 复制数据到设备（保持不变）
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // ======================= 测试行优化核函数 =======================
    {
        dim3 row_block(RC_TILE_SIZE * RC_TILE_SIZE); // 每个块RC_TILE_SIZE^2个线程
        dim3 row_grid((n + RC_TILE_SIZE - 1) / RC_TILE_SIZE);
        
        double row_start = getCurrentTime();
        row_optimized_matmul<<<row_grid, row_block>>>(d_C, d_A, d_B, n);
        cudaDeviceSynchronize();
        double row_end = getCurrentTime();
        printf("Row optimized kernel time: %.3f ms\n", row_end - row_start);
        cudaMemcpy(h_C_row, d_C, size, cudaMemcpyDeviceToHost);
        verifyResults(h_ref, h_C_row, n) ? printf("Row result: CORRECT\n") : printf("Row result: INCORRECT\n");
    }

    // ======================= 测试列优化核函数 =======================
    {
        dim3 col_block(RC_TILE_SIZE * RC_TILE_SIZE); // 每个块RC_TILE_SIZE^2个线程
        dim3 col_grid((n + RC_TILE_SIZE - 1) / RC_TILE_SIZE);
        
        double col_start = getCurrentTime();
        col_optimized_matmul<<<col_grid, col_block>>>(d_C, d_A, d_B, n);
        cudaDeviceSynchronize();
        double col_end = getCurrentTime();
        printf("Col optimized kernel time: %.3f ms\n", col_end - col_start);
        cudaMemcpy(h_C_col, d_C, size, cudaMemcpyDeviceToHost);
        verifyResults(h_ref, h_C_col, n) ? printf("Col result: CORRECT\n") : printf("Col result: INCORRECT\n");
    }

    // ===================== 测试块优化核函数 =====================
    {
        dim3 block_threads(BLOCK_SIZE, BLOCK_SIZE);
        int grid_dim = (n + BLOCK_SIZE * TILE_SIZE - 1) / (BLOCK_SIZE * TILE_SIZE);
        dim3 block_grid(grid_dim, grid_dim);
        
        double block_start = getCurrentTime();
        block_optimized_matmul<<<block_grid, block_threads>>>(d_C, d_A, d_B, n);
        cudaDeviceSynchronize();
        double block_end = getCurrentTime();
        printf("Block optimized kernel time: %.3f ms\n", block_end - block_start);
        cudaMemcpy(h_C_block, d_C, size, cudaMemcpyDeviceToHost);
        verifyResults(h_ref, h_C_block, n) ? printf("Block result: CORRECT\n") : printf("Block result: INCORRECT\n");
    }
    
    // 清理内存（保持不变）
    free(h_A); free(h_B); free(h_ref);
    free(h_C_row); free(h_C_col); free(h_C_block);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}

// ===================================================================
// 辅助函数实现
// ===================================================================

double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void initializeMatrix(float* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

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
// ===================================================================
// 核函数实现
// ===================================================================

// 参考实现（在GPU上运行用于验证）
__global__ void ref_matmul(float* out, float* A, float* B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

// ================== 寄存器优化的行划分实现 ==================
__global__ void row_optimized_matmul(float* out, const float* A, const float* B, int n) {
    __shared__ float sB[RC_TILE_SIZE][RC_TILE_SIZE];
    int row_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (row_idx >= n) return;
    
    for (int col = 0; col < n; col += RC_TILE_SIZE) {
        float C_part_row[RC_TILE_SIZE] = {0}; // 初始化为0
        int tile_end = min(col + RC_TILE_SIZE, n);
        int tile_width = tile_end - col;
        
        for (int tile_row = 0; tile_row < n; tile_row += RC_TILE_SIZE) {
            int tx = threadIdx.x;
            int row_end = min(tile_row + RC_TILE_SIZE, n);
            
            // 协作加载B的tile到共享内存
            for (int idx = tx; idx < RC_TILE_SIZE * RC_TILE_SIZE; idx += blockDim.x) {
                int i = idx / RC_TILE_SIZE;
                int j = idx % RC_TILE_SIZE;
                
                if (tile_row + i < n && col + j < n) {
                    sB[i][j] = B[(tile_row + i) * n + (col + j)];
                } else {
                    sB[i][j] = 0.0f;
                }
            }
            __syncthreads();
            
            // 加载A的一行部分
            float A_part_row[RC_TILE_SIZE] = {0};
            for (int i = 0; i < row_end - tile_row; i++) {
                A_part_row[i] = A[row_idx * n + (tile_row + i)];
            }
            
            // 计算部分结果
            for (int i = 0; i < tile_width; i++) {
                for (int k = 0; k < row_end - tile_row; k++) {
                    C_part_row[i] += A_part_row[k] * sB[k][i];
                }
            }
            __syncthreads();
        }
        
        // 存储结果
        for (int i = 0; i < tile_width; i++) {
            out[row_idx * n + (col + i)] = C_part_row[i];
        }
    }
}

// ================== 寄存器优化的列划分实现 ==================
__global__ void col_optimized_matmul(float* out, const float* A, const float* B, int n) {
    __shared__ float sA[RC_TILE_SIZE][RC_TILE_SIZE];
    int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (col_idx >= n) return;

    for (int row_start = 0; row_start < n; row_start += RC_TILE_SIZE) {
        float C_part_col[RC_TILE_SIZE] = {0};
        int row_end = min(row_start + RC_TILE_SIZE, n);
        int tile_height = row_end - row_start;
        
        for (int tile_col = 0; tile_col < n; tile_col += RC_TILE_SIZE) {
            int tx = threadIdx.x;
            int col_end = min(tile_col + RC_TILE_SIZE, n);
            
            // 协作加载A的tile到共享内存
            for (int idx = tx; idx < RC_TILE_SIZE * RC_TILE_SIZE; idx += blockDim.x) {
                int i = idx / RC_TILE_SIZE;
                int j = idx % RC_TILE_SIZE;
                
                if (row_start + i < n && tile_col + j < n) {
                    sA[i][j] = A[(row_start + i) * n + (tile_col + j)];
                } else {
                    sA[i][j] = 0.0f;
                }
            }
            __syncthreads();
            
            // 加载B的当前列部分
            float B_part_col[RC_TILE_SIZE] = {0};
            for (int i = 0; i < col_end - tile_col; i++) {
                if (tile_col + i < n) {
                    B_part_col[i] = B[(tile_col + i) * n + col_idx];
                }
            }
            
            // 计算部分结果
            for (int i = 0; i < tile_height; i++) {
                for (int j = 0; j < col_end - tile_col; j++) {
                    C_part_col[i] += sA[i][j] * B_part_col[j];
                }
            }
            __syncthreads();
        }
        
        // 存储结果
        for (int i = 0; i < tile_height; i++) {
            int out_row = row_start + i;
            if (out_row < n) {
                out[out_row * n + col_idx] = C_part_col[i];
            }
        }
    }
}

// ================== 寄存器优化的块划分实现 ==================
__global__ void block_optimized_matmul(float* out, const float* A, const float* B, int n) {
    // 共享内存声明
    __shared__ float sA[BLOCK_SIZE * TILE_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE * TILE_SIZE];
    
    // 线程在块内的局部坐标
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // 线程处理的全局输出块起始坐标
    int bx = blockIdx.x * (BLOCK_SIZE * TILE_SIZE);
    int by = blockIdx.y * (BLOCK_SIZE * TILE_SIZE);
    
    // 寄存器累加器
    float regC[TILE_SIZE][TILE_SIZE] = {{0.0f}};
    bool valid_thread = (bx + tx * TILE_SIZE < n) && (by + ty * TILE_SIZE < n);

    // 分块循环
    for (int t = 0; t < n; t += BLOCK_SIZE) {
        // 协作加载A的分块
        for (int i = 0; i < TILE_SIZE; i++) {
            int row = by + ty * TILE_SIZE + i;
            int col = t + tx;
            
            if (row < n && col < n) {
                sA[ty * TILE_SIZE + i][tx] = A[row * n + col];
            } else {
                sA[ty * TILE_SIZE + i][tx] = 0.0f;
            }
        }

        // 协作加载B的分块
        for (int i = 0; i < TILE_SIZE; i++) {
            int row = t + ty;
            int col = bx + tx * TILE_SIZE + i;
            
            if (row < n && col < n) {
                sB[ty][tx * TILE_SIZE + i] = B[row * n + col];
            } else {
                sB[ty][tx * TILE_SIZE + i] = 0.0f;
            }
        }
        
        __syncthreads();

        // 使用共享内存计算分块乘法
        for (int k = 0; k < BLOCK_SIZE && t + k < n; k++) {
            // 从共享内存加载到寄存器
            float regA[TILE_SIZE];
            for (int i = 0; i < TILE_SIZE; i++) {
                regA[i] = sA[ty * TILE_SIZE + i][k];
            }
            
            float regB[TILE_SIZE];
            for (int i = 0; i < TILE_SIZE; i++) {
                regB[i] = sB[k][tx * TILE_SIZE + i];
            }
            
            // 寄存器级乘累加
            for (int i = 0; i < TILE_SIZE; i++) {
                for (int j = 0; j < TILE_SIZE; j++) {
                    regC[i][j] += regA[i] * regB[j];
                }
            }
        }
        __syncthreads();
    }

    // 写回结果
    for (int i = 0; i < TILE_SIZE; i++) {
        for (int j = 0; j < TILE_SIZE; j++) {
            int row = by + ty * TILE_SIZE + i;
            int col = bx + tx * TILE_SIZE + j;
            if (row < n && col < n) {
                out[row * n + col] = regC[i][j];
            }
        }
    }
}