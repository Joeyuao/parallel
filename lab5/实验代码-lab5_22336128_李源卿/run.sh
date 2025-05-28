#!/bin/bash

# 编译程序
gcc -fopenmp -O3 main.c -o matmul

# 定义测试参数
THREADS=(1 2 4 8 16)
SIZES=(128 256 512 1024 2048)
SCHEDULE=1  # 默认调度类型

# 遍历线程数
for t in "${THREADS[@]}"; do
    echo "Threads num : $t"
    for s in "${SIZES[@]}"; do
        echo $s
        ./matmul $t $s $SCHEDULE
    done
    echo /n
done
