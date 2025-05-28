#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>

using namespace std;

void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn);
void cffti(int n, double w[]);
double cpu_time();
double ggl(double* ds);
void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn);

int main() {
    const int ln2_values[] = { 10, 12, 14, 16, 18, 20 };
    const int num_sizes = sizeof(ln2_values) / sizeof(int);

    cout << fixed << setprecision(3);
    cout << "         N      Time(s)\n";
    cout << "=======================\n";

    double seed = 331.0;

    for (int i = 0; i < num_sizes; ++i) {
        int ln2 = ln2_values[i];
        int n = 1 << ln2;

        double* w = new double[n];
        double* x = new double[2 * n];
        double* y = new double[2 * n];
        double* z = new double[2 * n];

        // 初始化数据
        for (int j = 0; j < 2 * n; j += 2) {
            x[j] = ggl(&seed);
            x[j + 1] = ggl(&seed);
            z[j] = x[j];
            z[j + 1] = x[j + 1];
        }

        cffti(n, w);  // 生成旋转因子

        // 正向+逆向FFT计时
        double start = cpu_time();
        cfft2(n, x, y, w, +1.0);   // 正向FFT
        cfft2(n, y, x, w, -1.0);   // 逆向FFT

        // 逆变换后缩放结果
        const double scale = 1.0 / n;
        for (int j = 0; j < 2 * n; ++j) {
            x[j] *= scale;
        }

        double elapsed = cpu_time() - start;

        // 误差计算
        double error = 0.0;
        for (int j = 0; j < 2 * n; j += 2) {
            error += pow(z[j] - x[j], 2) + pow(z[j + 1] - x[j + 1], 2);
        }
        error = sqrt(error / n);

        // 输出结果
        cout << "2^" << setw(2) << ln2 << " (" << setw(7) << n << ")"
            << "  " << fixed << setw(10) << setprecision(6) << elapsed << endl;

        delete[] w;
        delete[] x;
        delete[] y;
        delete[] z;
    }
    return 0;
}

// 修正旋转因子生成，虚部符号改为正
void cffti(int n, double w[]) {
    const double arg = 2.0 * M_PI / n;
    for (int i = 0; i < n / 2; ++i) {
        w[2 * i] = cos(i * arg);    // 实部正确
        w[2 * i + 1] = sin(i * arg);  // 修正虚部符号
    }
}

// 其余函数保持正确实现
void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn) {
    const int mj2 = 2 * mj;
    const int lj = n / mj2;

    for (int j = 0; j < lj; ++j) {
        const int jw = j * mj;
        const double wr = w[2 * jw];
        const double wi = sgn * w[2 * jw + 1];  // 应用符号参数

        for (int k = 0; k < mj; ++k) {
            const int idx = jw + k;
            // 蝶形计算
            c[2 * idx] = a[2 * idx] + b[2 * idx];
            c[2 * idx + 1] = a[2 * idx + 1] + b[2 * idx + 1];

            const double ambr = a[2 * idx] - b[2 * idx];
            const double ambi = a[2 * idx + 1] - b[2 * idx + 1];

            d[2 * idx] = wr * ambr - wi * ambi;
            d[2 * idx + 1] = wr * ambi + wi * ambr;
        }
    }
}

void ccopy(int n, double x[], double y[]) {
    for (int i = 0; i < 2 * n; ++i) y[i] = x[i];
}

void cfft2(int n, double x[], double y[], double w[], double sgn) {
    int m = (int)(log2(n));
    int mj = 1;
    int tgle = 1;

    step(n, mj, x, x + n, y, y + mj * 2, w, sgn);
    if (n == 2) return;

    for (int j = 0; j < m - 2; ++j) {
        mj *= 2;
        if (tgle) {
            step(n, mj, y, y + n, x, x + mj * 2, w, sgn);
            tgle = 0;
        }
        else {
            step(n, mj, x, x + n, y, y + mj * 2, w, sgn);
            tgle = 1;
        }
    }

    if (tgle) ccopy(n, y, x);
    mj = n / 2;
    step(n, mj, x, x + n, y, y + mj * 2, w, sgn);
}

double cpu_time() {
    return (double)clock() / CLOCKS_PER_SEC;
}

double ggl(double* ds) {
    double d = 2147483647.0;
    *ds = fmod(16807.0 * *ds, d);
    return (*ds - 1.0) / (d - 1.0);
}