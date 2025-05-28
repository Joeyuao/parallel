#!/bin/bash

# 脚本名称: run_matrix_test.sh
# 用途: 自动编译和运行矩阵乘法程序，测试不同矩阵大小和线程数组合

# 编译程序
echo "正在编译程序..."
gcc lab3_v1_sum_b.c -o sum_b -lpthread -lm
if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi
echo "编译成功。"

# 测试参数配置
arr_sizes=(1000000 4000000 16000000 64000000 128000000)  # 1M,4M,16M,64M,128M
thread_counts=( 2 4 8 16)   


# 模式1: 全面测试
for size in "${arr_sizes[@]}"; do
    
    for threads in "${thread_counts[@]}"; do
    echo "matrix_size:${size},threads_num: $threads"
        ./sum_b  $threads $size
    done
done

echo "所有测试完成。结果文件保存在 $output_dir 目录中"