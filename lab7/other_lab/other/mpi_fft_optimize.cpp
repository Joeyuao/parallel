#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <mpi.h>

using namespace std;

typedef struct { double r, i; } Complex;

void ccopy(int n, Complex x[], Complex y[]);
void cfft2(int n, Complex x[], Complex y[], Complex w[], double sgn,
    int rank, int size, MPI_Datatype mpi_complex_type);
void cffti(int n, Complex w[]);
void step(int n, int mj, Complex a[], Complex b[], Complex c[],
    Complex d[], Complex w[], double sgn);

double ggl(double* ds) {
    double d = 2147483647.0;
    *ds = fmod(16807.0 * *ds, d);
    return (*ds - 1.0) / (d - 1.0);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // �����Զ�����������
    MPI_Datatype MPI_COMPLEX_TYPE;
    MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_COMPLEX_TYPE);
    MPI_Type_commit(&MPI_COMPLEX_TYPE);

    // �����ģ����
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

        // �ؼ��޸�1��ȷ��n�ܱ�����������
        if (n % size != 0) {
            if (rank == 0) {
                cout << "Skip N=2^" << ln2 << " (" << n
                    << ") not divisible by " << size << endl;
            }
            continue;
        }

        Complex* x = nullptr, * z = nullptr, * w = nullptr;
        double total_time = 0.0;

        // ��ʼ������ (��rank 0)
        if (rank == 0) {
            x = new Complex[n];
            z = new Complex[n];
            w = new Complex[n / 2];

            double seed = 331.0;
            for (int j = 0; j < n; j++) {
                x[j].r = ggl(&seed);
                x[j].i = ggl(&seed);
                z[j] = x[j]; // ����ԭʼ����
            }
            cffti(n, w);
        }
        else {
            w = new Complex[n / 2]; // Ԥ�����ڴ�
        }

        // �ؼ��޸�2����ȷ�㲥nֵ
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // �㲥��ת����
        MPI_Bcast(w, n / 2, MPI_COMPLEX_TYPE, 0, MPI_COMM_WORLD);

        // ���䱾���ڴ�
        const int local_n = n / size;
        Complex* local_x = new Complex[local_n];
        Complex* local_y = new Complex[local_n];

        // ���ݷַ�
        MPI_Scatter(x, local_n, MPI_COMPLEX_TYPE,
            local_x, local_n, MPI_COMPLEX_TYPE,
            0, MPI_COMM_WORLD);

        // ��ʱ��ʼ
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        try {
            // FFT����
            cfft2(local_n, local_x, local_y, w, 1.0, rank, size, MPI_COMPLEX_TYPE);
            cfft2(local_n, local_y, local_x, w, -1.0, rank, size, MPI_COMPLEX_TYPE);
        }
        catch (...) {
            if (rank == 0) cerr << "Error in FFT computation" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // �ռ����
        MPI_Gather(local_x, local_n, MPI_COMPLEX_TYPE,
            x, local_n, MPI_COMPLEX_TYPE,
            0, MPI_COMM_WORLD);

        // ������
        double elapsed = MPI_Wtime() - start;

        // ��֤�����
        if (rank == 0) {
            double error = 0.0;
            const double scale = 1.0 / n;

            for (int j = 0; j < n; ++j) {
                double diff_r = z[j].r - x[j].r * scale;
                double diff_i = z[j].i - x[j].i * scale;
                error += diff_r * diff_r + diff_i * diff_i;
            }
            error = sqrt(error / n);

            cout << "N=2^" << ln2 << " (" << setw(7) << n << "):"
                << " Time=" << fixed << setw(12) << setprecision(6) << elapsed << "s" << endl; // ��λС��

            delete[] x;
            delete[] z;
        }

        delete[] w;
        delete[] local_x;
        delete[] local_y;
    }

    MPI_Type_free(&MPI_COMPLEX_TYPE);
    MPI_Finalize();
    return 0;
}

void cfft2(int n, Complex x[], Complex y[], Complex w[], double sgn,
    int rank, int size, MPI_Datatype mpi_complex_type) {
    const int m = static_cast<int>(log2(n));
    int mj = 1;
    int tgle = 1;

    // �ؼ��޸�3�����n>1У��
    if (n <= 1) {
        cerr << "Invalid FFT size: " << n << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int stage = 0; stage < m; ++stage) {
        step(n, mj,
            tgle ? x : y,
            (tgle ? x : y) + n / 2,
            tgle ? y : x,
            (tgle ? y : x) + mj,
            w, sgn);
        mj *= 2;
        tgle = !tgle;

        // �ؼ��޸�4����ֹ�������
        int comm_interval = static_cast<int>(log2(size));
        if (comm_interval <= 0) comm_interval = 1;

        if ((stage + 1) % comm_interval == 0) {
            MPI_Alltoall(tgle ? y : x,
                n / (size * mj > 0 ? size * mj : 1),  // ��ֹ����
                mpi_complex_type,
                tgle ? x : y,
                n / (size * mj > 0 ? size * mj : 1),
                mpi_complex_type,
                MPI_COMM_WORLD);
        }
    }
}

void cffti(int n, Complex w[]) {
    // �ؼ��޸�5����ֹn=0�����
    if (n <= 0) {
        cerr << "Invalid FFT size: " << n << endl;
        return;
    }

    const double arg = 2.0 * M_PI / n;
    for (int i = 0; i < n / 2; ++i) {
        w[i].r = cos(i * arg);
        w[i].i = sin(i * arg);
    }
}

void step(int n, int mj, Complex a[], Complex b[],
    Complex c[], Complex d[], Complex w[], double sgn) {
    const int mj2 = 2 * mj;
    // �ؼ��޸�6����ֹmjΪ��
    if (mj <= 0 || mj2 > n) return;

    const int lj = n / mj2;
    if (lj <= 0) return;

    for (int j = 0; j < lj; ++j) {
        const Complex wj = { w[j * mj].r, sgn * w[j * mj].i };
        for (int k = 0; k < mj; ++k) {
            const int idx = j * mj2 + k;

            if (idx >= n / 2) continue; // �߽籣��

            c[idx].r = a[idx].r + b[idx].r;
            c[idx].i = a[idx].i + b[idx].i;

            const double ambr = a[idx].r - b[idx].r;
            const double ambi = a[idx].i - b[idx].i;

            d[idx].r = wj.r * ambr - wj.i * ambi;
            d[idx].i = wj.i * ambr + wj.r * ambi;
        }
    }
}