#include <iostream>
#include <vector>
#include <random>
#include <chrono>
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

// 矩阵乘法
vector<vector<double>> matrixMultiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    int rows_B = B.size();
    int cols_B = B[0].size();

    // 检查矩阵维度是否可乘
    if (cols_A != rows_B) {
        cerr << "Error: Matrix dimensions do not match for multiplication!" << endl;
        return {};
    }

    // 初始化结果矩阵
    vector<vector<double>> result(rows_A, vector<double>(cols_B, 0.0));

    // 矩阵乘法计算
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            for (int k = 0; k < cols_A; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}

// 打印矩阵
void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

int main() {
    // 定义矩阵维度
    int rows_A = 1000, cols_A = 1000;
    int rows_B = 1000, cols_B = 1000;

    // 初始化矩阵 A 和 B
    vector<vector<double>> A, B;
    initializeMatrix(A, rows_A, cols_A);
    initializeMatrix(B, rows_B, cols_B);

    // 计算矩阵乘积
    auto start = high_resolution_clock::now();
    vector<vector<double>> C = matrixMultiply(A, B);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    // 输出运行时间
    cout << "Time taken: " << double(duration.count()/1000000.0) << " seconds" << endl;

    return 0;
}