#!/bin/bash

# 编译 OpenMP 程序（假设程序名为 omp_demo.c）
gcc -fopenmp omp_demo.c -o omp_demo

# 测试 1~8 线程的性能
for threads in {1,2,4,8,16}; do
    echo "===== 使用 $threads 个线程 ====="
    ./omp_demo $threads
done