#!/bin/bash

# 脚本名称: run_matrix_test.sh
# 用途: 自动编译和运行矩阵乘法程序，测试不同矩阵大小和线程数组合

# 编译程序
echo "正在编译程序..."
gcc lab3_v1_mat_b.c -o matrix_b -lpthread -lm
if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi
echo "编译成功。"

# 测试参数配置
# matrix_sizes=(128 256 512 1024 2048)      # 测试的矩阵维度(N×N)
matrix_sizes=(128 256 512 1024)
thread_counts=(1 2 4 8 16)          # 测试的线程数


# 模式1: 全面测试
for size in "${matrix_sizes[@]}"; do
    echo "matrix_size:${size}"
    for threads in "${thread_counts[@]}"; do
        ./matrix_b $size $threads
    done
done

echo "所有测试完成。结果文件保存在 $output_dir 目录中"