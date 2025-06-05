#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 0表示第一个GPU设备
    
    printf("============== GPU设备信息 ==============\n");
    printf("设备名称: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("全局内存: %.2f GB\n", (float)prop.totalGlobalMem/(1024 * 1024 * 1024));
    
    printf("\n============== 寄存器信息 ==============\n");
    printf("每个线程块的寄存器数量: %d\n", prop.regsPerBlock);
    printf("每个多处理器的寄存器数量: %d\n", prop.regsPerMultiprocessor);
    
    printf("\n============== 共享内存信息 ==============\n");
    printf("每个线程块的共享内存大小: %zu KB\n", prop.sharedMemPerBlock/1024);
    printf("每个多处理器的共享内存大小: %zu KB\n", prop.sharedMemPerMultiprocessor/1024);
    
    printf("\n============== 线程配置信息 ==============\n");
    printf("每个线程块的最大线程数: %d\n", prop.maxThreadsPerBlock);
    printf("每个多处理器的最大线程数: %d\n", prop.maxThreadsPerMultiProcessor);
    
    printf("\n============== 多维配置上限 ==============\n");
    printf("线程块维度上限: (%d, %d, %d)\n", 
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("网格维度上限: (%d, %d, %d)\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    printf("\n============== 其他关键信息 ==============\n");
    printf("多处理器数量: %d\n", prop.multiProcessorCount);
    printf("时钟频率: %.2f GHz\n", prop.clockRate/1e6);
    printf("内存时钟频率: %.2f GHz\n", prop.memoryClockRate/1e6);
    printf("内存总线宽度: %d-bit\n", prop.memoryBusWidth);
    
    // CUDA 10.0中没有L2缓存大小字段
    // printf("L2缓存大小: %d KB\n", prop.l2CacheSize/1024);
    
    return 0;
}