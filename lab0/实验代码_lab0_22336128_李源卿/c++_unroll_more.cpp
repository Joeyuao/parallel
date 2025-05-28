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
        // 四路j展开：每次处理4列
        for (; j <= cols_B - 4; j += 4) {
            double sum1 = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
            int k = 0;
            // 八路k展开：每次处理8个元素
            for (; k <= cols_A - 8; k += 8) {
                // 预加载A的连续8个元素
                const double a0 = A[i][k];
                const double a1 = A[i][k+1];
                const double a2 = A[i][k+2];
                const double a3 = A[i][k+3];
                const double a4 = A[i][k+4];
                const double a5 = A[i][k+5];
                const double a6 = A[i][k+6];
                const double a7 = A[i][k+7];

                // 为四列分别累加乘积结果
                sum1 += a0*B[k][j]   + a1*B[k+1][j] + a2*B[k+2][j] + a3*B[k+3][j] 
                      + a4*B[k+4][j] + a5*B[k+5][j] + a6*B[k+6][j] + a7*B[k+7][j];
                
                sum2 += a0*B[k][j+1]   + a1*B[k+1][j+1] + a2*B[k+2][j+1] + a3*B[k+3][j+1]
                      + a4*B[k+4][j+1] + a5*B[k+5][j+1] + a6*B[k+6][j+1] + a7*B[k+7][j+1];
                
                sum3 += a0*B[k][j+2]   + a1*B[k+1][j+2] + a2*B[k+2][j+2] + a3*B[k+3][j+2]
                      + a4*B[k+4][j+2] + a5*B[k+5][j+2] + a6*B[k+6][j+2] + a7*B[k+7][j+2];
                
                sum4 += a0*B[k][j+3]   + a1*B[k+1][j+3] + a2*B[k+2][j+3] + a3*B[k+3][j+3]
                      + a4*B[k+4][j+3] + a5*B[k+5][j+3] + a6*B[k+6][j+3] + a7*B[k+7][j+3];
            }
            // 处理剩余k元素（0-7个）
            for (; k < cols_A; ++k) {
                const double a = A[i][k];
                sum1 += a * B[k][j];
                sum2 += a * B[k][j+1];
                sum3 += a * B[k][j+2];
                sum4 += a * B[k][j+3];
            }
            result[i][j]   = sum1;
            result[i][j+1] = sum2;
            result[i][j+2] = sum3;
            result[i][j+3] = sum4;
        }
        // 处理剩余j列（0-3列）
        for (; j < cols_B; ++j) {
            double sum = 0.0;
            int k = 0;
            // 八路k展开
            for (; k <= cols_A - 8; k += 8) {
                sum += A[i][k]*B[k][j]   + A[i][k+1]*B[k+1][j]
                     + A[i][k+2]*B[k+2][j] + A[i][k+3]*B[k+3][j]
                     + A[i][k+4]*B[k+4][j] + A[i][k+5]*B[k+5][j]
                     + A[i][k+6]*B[k+6][j] + A[i][k+7]*B[k+7][j];
            }
            // 处理剩余k元素
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