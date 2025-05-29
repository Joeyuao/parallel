// #include <__clang_cuda_builtin_vars.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#define BDIM 16
__global__ void trans(int* out, int* in, int n) {
    __shared__ int smem[BDIM * BDIM];

    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    smem[ty*BDIM + tx] = in[(by+ty)*n + tx+bx];
    __syncthreads();
    out[(bx+ty)*n + by+tx] = smem[tx*BDIM + ty];
}
__global__ void trans_solve_confilct(int* out, int* in, int n) {
    __shared__ int smem[BDIM * BDIM];

    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    smem[ty*BDIM + tx] = in[(by+ty)*n + tx+bx];
    __syncthreads();
    out[(bx+tx)*n + by+ty] = smem[ty*BDIM + tx];
}
void initializeMatrix(int* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i*n + j] = i * n + j;  // Simple initialization pattern
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

int main() {
    const int n = 32 * 16;  // blockDim.x * BlockDim.x = 32 * 16 = 512
    size_t size = n * n * sizeof(int);
    
    // Allocate and initialize host memory
    int* h_in = (int*)malloc(size);
    int* h_out = (int*)malloc(size);
    initializeMatrix(h_in, n);
    
    // Allocate device memory
    int *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    
    // Copy data to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 ThreadDim(BDIM, BDIM);
    dim3 BlockDim(32, 32);
    trans_solve_confilct<<<BlockDim, ThreadDim>>>(d_out, d_in, n);
    
    // Copy result back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    // Print a small portion for verification
    printf("Original matrix (top-left 5x5):\n");
    printMatrix(h_in, 5, n);
    printf("\nTransposed matrix (top-left 5x5):\n");
    printMatrix(h_out, 5, n);
    
    // Cleanup
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    
    cudaDeviceSynchronize();
    return 0;
}