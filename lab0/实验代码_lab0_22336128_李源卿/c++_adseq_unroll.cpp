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

    if (cols_A != rows_B) {
        cerr << "Error: Matrix dimensions do not match for multiplication!" << endl;
        return {};
    }

    vector<vector<double>> result(rows_A, vector<double>(cols_B, 0.0));

    // 优化后的循环展开策略：i -> k -> j（j循环展开4次）
    for (int i = 0; i < rows_A; ++i) {
        for (int k = 0; k < cols_A; ++k) {
            const double a = A[i][k];  // 缓存A[i][k]，减少重复访问
            int j = 0;

            // 主循环：每次处理4个j值（4路展开）
            for (; j <= cols_B - 4; j += 4) {
                // 预加载B的4个连续元素地址（行优先存储，内存连续）
                const double* b_ptr = &B[k][j];
                const double b0 = b_ptr[0], b1 = b_ptr[1], b2 = b_ptr[2], b3 = b_ptr[3];

                // 直接累加到result，减少中间变量
                result[i][j]   += a * b0;
                result[i][j+1] += a * b1;
                result[i][j+2] += a * b2;
                result[i][j+3] += a * b3;
            }

            // 处理剩余不足4个的j值
            for (; j < cols_B; ++j) {
                result[i][j] += a * B[k][j];
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