#!/bin/bash

# 编译 OpenMP 程序（假设程序名为 omp_demo.c）
g++ -fopenmp parallel_ws.cpp -o parallel_ws

# 测试 1~8 线程的性能
for threads in {1,2,4,8,16}; do
    echo "===== 使用 $threads 个线程 ====="
    ./parallel_ws $threads
done