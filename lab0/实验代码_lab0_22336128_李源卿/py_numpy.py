import numpy as np
import time
# 生成随机矩阵 A (m×k) 和 B (k×n)
m, k, n = 4,4 , 4  # 定义矩阵维度（可修改为任意大小）

A=[[i+j*4+1 for i in range(0,4)] for j in range(0,4)]
B=[[i+j*4+1 for i in range(0,4)] for j in range(0,4)]
A=np.array(A)
B=np.array(B)
start=time.time()
C = np.dot(A, B)           # 或者 C = A @ B（Python 3.5+ 支持）
end=time.time()
print(C)
C=[[0 for i in range(0,4)] for j in range(0,4)]
B=B.T
for i in range(4):
    for j in range(4):
        for k in range(4):
            C[i][j]+=A[i][k]*B[i][k]
print(C)
print("time cost:",end-start)