// #include <__clang_cuda_builtin_vars.h>
#include <stdio.h>
#include <cuda_runtime.h>

// 2D CUDA kernel function
__global__ void hello_world_kernel() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    printf("Hello World from thread (%d, %d) in Block (%d)\n", threadIdx.x, threadIdx.y, blockIdx.x);
}

int main() {
    // Threads per block in each dimension
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    
    // Number of blocks in each dimension
    dim3 blocksInGrid(16);       // 4x4 = 16 blocks
    
    // Launch the 2D kernel
    hello_world_kernel<<<blocksInGrid, threadsPerBlock>>>();
    
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "hello_world_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    
    // cudaDeviceReset must be called before exiting
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    
    return 0;
}