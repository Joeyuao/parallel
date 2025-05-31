#include <iostream>
#include <vector>
#include <random>
#include <chrono>
using namespace std;
using namespace std::chrono;

random_device rd;  
mt19937 gen(rd());
uniform_real_distribution<> dis(0.0, 1.0);

void initializeMatrix(vector<vector<double>>& matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        vector<double> row(cols);
        for (int j = 0; j < cols; ++j) {
            row[j] = dis(gen);
        }
        matrix.push_back(row);
    }
}

vector<vector<double>> matrixMultiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int rows_A = A.size();
    int cols_A = A[0].size();
    int cols_B = B[0].size();

    if (cols_A != B.size()) {
        cerr << "Error: Matrix dimensions do not match for multiplication!" << endl;
        return {};
    }

    vector<vector<double>> result(rows_A, vector<double>(cols_B, 0.0));

    for (int i = 0; i < rows_A; ++i) {
        int j = 0;
        // 每次处理2个j（列）以减少循环次数
        for (; j <= cols_B - 2; j += 2) {
            double sum1 = 0.0, sum2 = 0.0;
            int k = 0;
            // 每次处理4个k（展开因子=4）
            for (; k <= cols_A - 4; k += 4) {
                // 预加载A的元素
                const double a0 = A[i][k];
                const double a1 = A[i][k+1];
                const double a2 = A[i][k+2];
                const double a3 = A[i][k+3];

                // 为两个不同的j值计算乘积并累加
                sum1 += a0 * B[k][j] + a1 * B[k+1][j] + a2 * B[k+2][j] + a3 * B[k+3][j];
                sum2 += a0 * B[k][j+1] + a1 * B[k+1][j+1] + a2 * B[k+2][j+1] + a3 * B[k+3][j+1];
            }
            // 处理剩余k值
            for (; k < cols_A; ++k) {
                const double a = A[i][k];
                sum1 += a * B[k][j];
                sum2 += a * B[k][j+1];
            }
            result[i][j] = sum1;
            result[i][j+1] = sum2;
        }
        // 处理剩余j值
        for (; j < cols_B; ++j) {
            double sum = 0.0;
            int k = 0;
            // 同样展开k循环
            for (; k <= cols_A - 4; k += 4) {
                sum += A[i][k] * B[k][j] + A[i][k+1] * B[k+1][j] +
                       A[i][k+2] * B[k+2][j] + A[i][k+3] * B[k+3][j];
            }
            // 处理剩余k值
            for (; k < cols_A; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

void printMatrix(const vector<vector<double>>& matrix) {
    for (const auto& row : matrix) {
        for (double val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

int main() {
    int rows_A = 1000, cols_A = 1000;
    int rows_B = 1000, cols_B = 1000;

    vector<vector<double>> A, B;
    initializeMatrix(A, rows_A, cols_A);
    initializeMatrix(B, rows_B, cols_B);

    auto start = high_resolution_clock::now();
    vector<vector<double>> C = matrixMultiply(A, B);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    cout << "Time taken: " << double(duration.count()/1000000.0) << " seconds" << endl;

    return 0;
}