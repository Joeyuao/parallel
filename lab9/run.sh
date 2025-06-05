#!/bin/bash

# 编译CUDA程序
nvcc main.cu -o main

MATRIX_SIZES=(512 1024 2048)
BLOCK_SIZES=(8 16 32)

for size in "${MATRIX_SIZES[@]}"; do
    for block in "${BLOCK_SIZES[@]}"; do
        ./main $size $block
    done
done