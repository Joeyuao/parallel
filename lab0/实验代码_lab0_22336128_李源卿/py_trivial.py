import time
import random
m,k,n=1000,1000,1000
def matrix_multiply(A, B):
    """
    矩阵乘法
    """
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    if cols_A != rows_B:
        raise ValueError("矩阵A的列数必须等于矩阵B的行数")
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):          
        for j in range(cols_B):      
            for k in range(cols_A):  
                result[i][j] += A[i][k] * B[k][j]
    return result
def generate_random_matrix(rows, cols):
    """
    生成随机矩阵
    """
    return [[random.random() for _ in range(cols)] for _ in range(rows)]
if __name__ == "__main__":
    A=generate_random_matrix(m,k)
    B=generate_random_matrix(k,n)
    start=time.time()
    C = matrix_multiply(A, B)
    end=time.time()
    print(f"time cost:{end-start}")