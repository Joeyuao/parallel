#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <mpi.h>

using namespace std;

void ccopy(int n, double x[], double y[]);
void cfft2(int n, double x[], double y[], double w[], double sgn, int rank, int size);
void cffti(int n, double w[]);
double cpu_time();
double ggl(double* ds);
void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int ln2_values[] = { 10, 12, 14, 16, 18, 20 };
    const int num_sizes = sizeof(ln2_values) / sizeof(int);

    if (rank == 0) {
        cout << "==========================================\n";
        cout << "  MPI Processes: " << size << "\n";
        cout << "==========================================\n";
    }

    for (int i = 0; i < num_sizes; ++i) {
        int ln2 = ln2_values[i];
        int n = 1 << ln2;

        // Check divisibility
        if (n % size != 0) {
            if (rank == 0) {
                cerr << "Error: N=2^" << ln2 << " (" << n
                    << ") is not divisible by " << size << " processes\n";
            }
            continue;
        }

        double* x = nullptr, * z = nullptr, * w = nullptr;
        double total_time = 0.0;

        if (rank == 0) {
            w = new double[n];
            cffti(n, w);

            x = new double[2 * n];
            z = new double[2 * n];

            double seed = 331.0;
            for (int j = 0; j < 2 * n; j += 2) {
                x[j] = ggl(&seed);
                x[j + 1] = ggl(&seed);
                z[j] = x[j];
                z[j + 1] = x[j + 1];
            }
        }
        else {
            w = new double[n];
        }

        // Broadcast twiddle factors
        MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        int local_n = n / size;
        double* local_x = new double[2 * local_n];
        double* local_y = new double[2 * local_n];

        // Timing start
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        // Scatter data
        MPI_Scatter(x, 2 * local_n, MPI_DOUBLE,
            local_x, 2 * local_n, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        // FFT computation
        cfft2(local_n, local_x, local_y, w, 1.0, rank, size);
        cfft2(local_n, local_y, local_x, w, -1.0, rank, size);

        // Gather results
        MPI_Gather(local_x, 2 * local_n, MPI_DOUBLE,
            x, 2 * local_n, MPI_DOUBLE,
            0, MPI_COMM_WORLD);

        // Timing end
        double elapsed = MPI_Wtime() - start;

        // Verification and output
        if (rank == 0) {
            double error = 0.0;
            double fnm1 = 1.0 / n;
            for (int j = 0; j < 2 * n; j += 2) {
                error += pow(z[j] - fnm1 * x[j], 2) +
                    pow(z[j + 1] - fnm1 * x[j + 1], 2);
            }
            error = sqrt(fnm1 * error);

            cout << "N=2^" << ln2 << " (" << setw(7) << n << "):"
                << " Time=" << fixed << setw(12) << setprecision(6) << elapsed << "s" <<setw(12)<<error<< endl; // ���ӿ��ȱ�֤����
            delete[] x;
            delete[] z;
        }

        delete[] w;
        delete[] local_x;
        delete[] local_y;
    }

    MPI_Finalize();
    return 0;
}


void cfft2(int n, double x[], double y[], double w[], double sgn, int rank, int size) {
    int m = (int)(log(n) / log(2));
    int mj = 1;
    int tgle = 1;

    step(n, mj, &x[0], &x[n], &y[0], &y[mj], w, sgn);
    if (n == 2) return;

    for (int j = 0; j < m - 2; j++) {
        mj *= 2;
        if (tgle) {
            step(n, mj, &y[0], &y[n], &x[0], &x[mj], w, sgn);
            tgle = 0;
        }
        else {
            step(n, mj, &x[0], &x[n], &y[0], &y[mj], w, sgn);
            tgle = 1;
        }
    }

    if (tgle) ccopy(n, y, x);
    mj = n / 2;
    step(n, mj, &x[0], &x[n], &y[0], &y[mj], w, sgn);

    MPI_Alltoall(x, 2 * (n / size), MPI_DOUBLE,
        y, 2 * (n / size), MPI_DOUBLE, MPI_COMM_WORLD);
    ccopy(n, y, x);
}

void ccopy(int n, double x[], double y[]) {
    for (int i = 0; i < 2 * n; ++i) y[i] = x[i];
}

void cffti(int n, double w[]) {
    double arg = 2.0 * M_PI / n;
    for (int i = 0; i < n / 2; ++i) {
        w[2 * i] = cos(i * arg);
        w[2 * i + 1] = sin(i * arg);
    }
}

double ggl(double* ds) {
    double d = 2147483647.0;
    *ds = fmod(16807.0 * *ds, d);
    return (*ds - 1.0) / (d - 1.0);
}

void step(int n, int mj, double a[], double b[], double c[], double d[], double w[], double sgn) {
    int mj2 = 2 * mj;
    int lj = n / mj2;

    for (int j = 0; j < lj; j++) {
        int jw = j * mj;
        double wr = w[2 * jw];
        double wi = sgn * w[2 * jw + 1];

        for (int k = 0; k < mj; k++) {
            int idx = jw + k;
            c[2 * idx] = a[2 * idx] + b[2 * idx];
            c[2 * idx + 1] = a[2 * idx + 1] + b[2 * idx + 1];

            double ambr = a[2 * idx] - b[2 * idx];
            double ambu = a[2 * idx + 1] - b[2 * idx + 1];

            d[2 * idx] = wr * ambr - wi * ambu;
            d[2 * idx + 1] = wi * ambr + wr * ambu;
        }
    }
}