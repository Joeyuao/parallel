#!/bin/bash

# 脚本名称: run_matrix_test.sh
# 用途: 自动编译和运行矩阵乘法程序，测试不同参数组合

# 编译程序
echo "Compiling the program..."
gcc lab3_v2_mat_bc.c -o mat_bc -lpthread -lm
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful."

# 测试参数组合
matrix_sizes=(128 256 512 1024 2048)
block_percents=(16 8 4 2 1)  # BLOCK_PERCENT 参数

# 创建结果目录
mkdir -p test_results

# 运行测试
for size in "${matrix_sizes[@]}"; do
    for percent in "${block_percents[@]}"; do
        echo "Testing: Matrix Size=$size, Threads=4, Block Percent=$percent"
        
        # 运行程序并保存输出
        ./mat_bc $size 4 $percent

    done
done

echo "All tests completed. Results are saved in the test_results directory."