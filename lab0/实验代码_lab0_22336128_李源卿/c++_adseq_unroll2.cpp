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

    const int I_UNROLL = 2;  // i方向展开因子
    const int J_UNROLL = 4;  // j方向展开因子

    // 主循环：每次处理2行(i方向)和4列(j方向)
    for (int i = 0; i < rows_A; i += I_UNROLL) {
        // 边界处理：最后不足展开因子的部分
        int i_end = std::min(i + I_UNROLL, rows_A);
        
        for (int k = 0; k < cols_A; ++k) {
            // 预加载A的两行元素
            double a0 = (i < rows_A) ? A[i][k] : 0.0;
            double a1 = (i+1 < rows_A) ? A[i+1][k] : 0.0;

            // j方向展开处理
            int j = 0;
            for (; j <= cols_B - J_UNROLL; j += J_UNROLL) {
                // 同时处理i的两行和j的四列
                for (int ii = i; ii < i_end; ++ii) {
                    double* result_row = result[ii].data();
                    const double* B_row = B[k].data();
                    
                    // 使用局部变量暂存中间结果
                    double r0 = result_row[j];
                    double r1 = result_row[j+1];
                    double r2 = result_row[j+2];
                    double r3 = result_row[j+3];

                    // 计算累加值
                    double a = (ii == i) ? a0 : a1;
                    r0 += a * B_row[j];
                    r1 += a * B_row[j+1];
                    r2 += a * B_row[j+2];
                    r3 += a * B_row[j+3];

                    // 写回结果
                    result_row[j]   = r0;
                    result_row[j+1] = r1;
                    result_row[j+2] = r2;
                    result_row[j+3] = r3;
                }
            }

            // 处理剩余列
            for (; j < cols_B; ++j) {
                const double B_val = B[k][j];
                if (i < rows_A)   result[i][j]   += a0 * B_val;
                if (i+1 < rows_A) result[i+1][j] += a1 * B_val;
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