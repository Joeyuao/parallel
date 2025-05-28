#!/bin/bash
# 文件名: run_pi_benchmark.sh
# 用途: 自动测试不同线程数和点数量的组合

# 编译程序
echo "Compiling program..."
gcc monte_carlo1.c -o monte_carlo1 -lpthread -lm -I.
if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 清空结果文件
echo "points,threads,efficiency,error,serial_time,para_time,speedup" > pi_results1.txt

# 测试参数范围
MIN_POINTS=1024
MAX_POINTS=65536
POINTS_STEP=1024

MIN_THREADS=1
MAX_THREADS=16


for (( points=$MIN_POINTS; points<=$MAX_POINTS; points+=$POINTS_STEP ))
do
    for (( threads=$MIN_THREADS; threads<=$MAX_THREADS; threads++ ))
    do
        echo "Running with $points points and $threads threads..."
        ./monte_carlo1 $threads $points 
    done
done


# 清理临时文件
rm raw_output.txt

echo "Benchmark completed! Results saved to pi_results1.txt"