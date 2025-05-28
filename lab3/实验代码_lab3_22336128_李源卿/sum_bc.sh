#!/bin/bash

# 脚本名称: run_sum_test.sh
# 用途: 自动编译和运行数组求和程序，测试不同参数组合

# 编译程序
echo "Compiling the program..."
gcc lab3_v2_sum_bc.c -o sum_bc -lpthread
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi
echo "Compilation successful."

# 测试参数组合
array_sizes=(1000000 4000000 16000000 64000000 128000000)  # 1M,4M,16M,64M,128M

block_percents=(16 8 4 2 1)               # BLOCK_PERCENT 参数

# 创建结果目录
mkdir -p sum_test_results

# 运行测试
for size in "${array_sizes[@]}"; do
echo "Testing: Array Size=$size, Threads=4, Block Percent="${block_percents[@]}""
    for percent in "${block_percents[@]}"; do
        
        # 运行程序并保存输出
        ./sum_bc 4 $size $percent
        
    done
done

echo "All tests completed. Results are saved in the sum_test_results directory."