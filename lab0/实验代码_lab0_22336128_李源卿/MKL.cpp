#include <iostream>
#include <mkl.h> // 包含 Intel MKL 头文件
#include <chrono>
#include <random>
#include<ctime>
using namespace std;
using namespace std::chrono;
random_device rd;  // 随机种子
mt19937 gen(rd()); // 随机数引擎
uniform_real_distribution<> dis(0.0, 1.0); // 均匀分布 [0, 1)
// 初始化矩阵
void initializeMatrix(vector<vector<double>>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        vector<double> row(cols);
        for (int j = 0; j < cols; ++j) {
            row[j] = dis(gen); // 随机初始化 [0, 1) 之间的浮点数
        }
        matrix.push_back(row);
    }
}
int main() {
    // 定义矩阵维度
    const int m = 1000; // 矩阵 A 的行数
    const int k = 1000; // 矩阵 A 的列数，矩阵 B 的行数
    const int n = 1000; // 矩阵 B 的列数

    // 分配内存并初始化矩阵 A (m x k) 和 B (k x n)
    double *A = (double *)mkl_malloc(m * k * sizeof(double), 64);
    double *B = (double *)mkl_malloc(k * n * sizeof(double), 64);
    double *C = (double *)mkl_malloc(m * n * sizeof(double), 64); // 结果矩阵 C (m x n)

    if (A == nullptr || B == nullptr || C == nullptr) {
        std::cerr << "Error: Memory allocation failed!" << std::endl;
        return 1;
    }

    // 初始化矩阵 A 和 B
    for (int i = 0; i < m * k; ++i) A[i] = dis(gen); // A = [1, 2; 3, 4; 5, 6]
    for (int i = 0; i < k * n; ++i) B[i] = dis(gen); // B = [1, 2, 3, 4; 5, 6, 7, 8]

   

    // 调用 MKL 的矩阵乘法函数 cblas_dgemm
    const double alpha = 1.0; // 系数 alpha
    const double beta = 0.0;  // 系数 beta
    auto start = high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout<<double(duration.count())/1000000.0<<"秒"<<endl;
    // // 打印结果矩阵 C
    // std::cout << "\nMatrix C (A * B):" << std::endl;
    // for (int i = 0; i < m; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         std::cout << C[i * n + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // 释放内存
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}