#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include <mpi.h>

using namespace std;

void ccopy ( int n, double x[], double y[] );
int cfft2 ( int n, double x[], double y[], double w[], double sgn );
void cffti ( int n, double w[] );
double cpu_time ( void );
int parallel_cfft2 ( int n, double x[], double y[], double w[], double sgn );
void parallel_step(int n, int mj, double a[], double b[], double c[],
    double d[], double w[], double sgn);
double ggl ( double *ds );
void step ( int n, int mj, double a[], double b[], double c[], double d[], 
    double w[], double sgn );
void timestamp ( );
void print_arr(int n,double *nums){
	std::cout<<"print nums:"<<endl;
	for(int i=0; i < n; i++){
		std::cout<<nums[i]<<' ';
	}
	std::cout<<endl;
}
int main (int argc, char** argv)
{
	MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double stime;
    double stime1;
    double ptime1;
	double ptime;
    double error;
    int first;
    double flops;
    double fnm1;
    int i;
    int icase;
    int it;
    int ln2;
    double mflops;
    int n;
    int nits = 10000;
    static double seed;
    double sgn;
    double *w;
    double *x;
    double *y;
    double *z;
    double z0;
    double z1;
    double *para_w;
    double *para_x;
    double *para_y;
    double *para_z;
	if (rank == 0) {
        std::cout << "==========================================\n";
        std::cout << "  MPI Processes: " << size << "\n";
        std::cout << "==========================================\n";
    }
    
    seed  = 331.0;
    n = 128;
    for ( ln2 = 4; ln2 <= 4; ln2++ )
    {
        // n = 2 * n;
        w = new double[  n];
        x = new double[2*n];
        y = new double[2*n];
		para_w = new double[  n];
        para_x = new double[2*n];
        para_y = new double[2*n];
		if ( rank == 0 )
		{
			for ( i = 0; i < 2 * n; i = i + 2 )
			{
				z0 = ggl ( &seed );
				z1 = ggl ( &seed );
				para_x[i] = x[i] = z0;
				para_x[i+1] = x[i+1] = z1;
			}
		} 
		MPI_Bcast(x, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(para_x, 2*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if(rank == 0){
			cffti ( n, w );
			// cffti ( n, para_w);
		}
		MPI_Bcast(w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		// MPI_Bcast(para_w, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		stime1 = MPI_Wtime();
		sgn = + 1.0;

        // serial
		if(rank == 0){
			cfft2 ( n, x, y, w, sgn );
            // print_arr(n,y);
			stime = MPI_Wtime() - stime1;
			std::cout<<"serial time:"<<stime<<endl;
		}
	
		MPI_Barrier(MPI_COMM_WORLD);

		ptime1 = MPI_Wtime();
	    parallel_cfft2(n, para_x, para_y, w, sgn);
		
        ptime = MPI_Wtime() - ptime1;
		if (rank == 0)
		{
			for ( i = 0; i < 2 * n; i = i + 2 )
            {
                error = error 
                + pow ( para_y[i]   - y[i], 2 )
                + pow ( para_y[i+1] - y[i+1], 2 );
            }
            error = sqrt ( error / double(n) );
            std::cout << "  " << setw(12) << n
                    << "  " << setw(8) << nits
                    << "  " << setw(12) << error<<endl;
		}
        delete [] w;
        delete [] x;
        delete [] y;
		delete [] para_w;
        delete [] para_x;
        delete [] para_y;

        if(rank == 0){
            std::cout << "\n";
            std::cout << "FFT_SERIAL:\n";
            std::cout << "  Normal end of execution.\n";
            std::cout << "\n";
            timestamp ( );
        }
    
	
    }
    std::cout<<"there is end! "<<"my rank:"<<rank<<endl;
    MPI_Finalize();
    
    return 0;
}
//****************************************************************************80

void ccopy ( int n, double x[], double y[] )
{
    int i;

    for ( i = 0; i < n; i++ )
    {
        y[i*2+0] = x[i*2+0];
        y[i*2+1] = x[i*2+1];
    }
    return;
}
//****************************************************************************80
int parallel_cfft2 ( int n, double x[], double y[], double w[], double sgn )
{
    int j;
    int m;
    int mj;
    int tgle;

    m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) );
    mj   = 1;
    tgle = 1;
    parallel_step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );

    if ( n == 2 )
    {
        return mj;
    }

    for ( j = 0; j < m - 2; j++ )
    {
        mj = mj * 2;
        if ( tgle )
        {
            parallel_step ( n, mj, &y[0*2+0], &y[(n/2)*2+0], &x[0*2+0], &x[mj*2+0], w, sgn );
            tgle = 0;
        }
        else
        {
            parallel_step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
            tgle = 1;
        }
    }
    if ( tgle ) 
    {
        ccopy ( n, y, x );
    }

    mj = n / 2;
    parallel_step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );

    return mj;
}
int cfft2 ( int n, double x[], double y[], double w[], double sgn )
{
    int j;
    int m;
    int mj;
    int tgle;

    m = ( int ) ( log ( ( double ) n ) / log ( 1.99 ) );
    mj   = 1;
    tgle = 1;
    step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );

    if ( n == 2 )
    {
        return mj;
    }

    for ( j = 0; j < m - 2; j++ )
    {
        mj = mj * 2;
        if ( tgle )
        {
            step ( n, mj, &y[0*2+0], &y[(n/2)*2+0], &x[0*2+0], &x[mj*2+0], w, sgn );
            tgle = 0;
        }
        else
        {
            step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );
            tgle = 1;
        }
    }
    if ( tgle ) 
    {
        ccopy ( n, y, x );
    }

    mj = n / 2;
    step ( n, mj, &x[0*2+0], &x[(n/2)*2+0], &y[0*2+0], &y[mj*2+0], w, sgn );

    return mj;
}
//****************************************************************************80

void cffti ( int n, double w[] )
{
    double arg;
    double aw;
    int i;
    int n2;
    const double pi = 3.141592653589793;

    n2 = n / 2;
    aw = 2.0 * pi / ( ( double ) n );

    for ( i = 0; i < n2; i++ )
    {
        arg = aw * ( ( double ) i );
        w[i*2+0] = cos ( arg );
        w[i*2+1] = sin ( arg );
    }
    return;
}
//****************************************************************************80

double cpu_time ( void )
{
    double value;

    value = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC;

    return value;
}
//****************************************************************************80

double ggl ( double *seed )
{
    double d2 = 0.2147483647e10;
    double t;
    double value;

    t = *seed;
    t = fmod ( 16807.0 * t, d2 );
    *seed = t;
    value = ( t - 1.0 ) / ( d2 - 1.0 );

    return value;
}
//****************************************************************************80

void parallel_step(int n, int mj, double a[], double b[], double c[],
                   double d[], double w[], double sgn) 
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int mj2 = 2 * mj;
    const int lj = n / mj2;  // lj must be divisible by size
    const int chunk = lj / size;
    const int j_start = rank * chunk;
    const int j_end = j_start + chunk;

    // 1. 本地计算
    for (int j = j_start; j < j_end; ++j) {
        const int jw = j * mj;
        const int jc = j * mj2;
        const int jd = jc;

        double wjw[2] = {w[jw*2], w[jw*2+1]};
        if (sgn < 0.0) wjw[1] = -wjw[1];

        for (int k = 0; k < mj; ++k) {
            const int idx_a = (jw + k) * 2;
            const int idx_c = (jc + k) * 2;
            const int idx_d = (jd + k) * 2;

            c[idx_c]   = a[idx_a] + b[idx_a];
            c[idx_c+1] = a[idx_a+1] + b[idx_a+1];
            
            const double ambr = a[idx_a] - b[idx_a];
            const double ambu = a[idx_a+1] - b[idx_a+1];
            d[idx_d]   = wjw[0] * ambr - wjw[1] * ambu;
            d[idx_d+1] = wjw[1] * ambr + wjw[0] * ambu;
        }
    }

    // 2. 准备通信数据
    const int elements_per_chunk = 4 * mj * chunk; // 4 = 2 (complex) * 2 (c+d)
    int send_counts[size], recv_counts[size];
    int sdispls[size], rdispls[size];
    
    for (int p = 0; p < size; ++p) {
        send_counts[p] = elements_per_chunk;
        recv_counts[p] = elements_per_chunk;
        sdispls[p] = p * elements_per_chunk;
        rdispls[p] = p * elements_per_chunk;
    }

    // 3. 使用MPI_Pack打包数据
    int pack_size;
    MPI_Pack_size(elements_per_chunk, MPI_DOUBLE, MPI_COMM_WORLD, &pack_size);
    char* sendbuf = new char[pack_size];
    char* recvbuf = new char[size * pack_size];
    
    int position = 0;
    for (int j = j_start; j < j_end; ++j) {
        const int jc = j * mj2;
        for (int k = 0; k < mj; ++k) {
            MPI_Pack(&c[(jc+k)*2], 2, MPI_DOUBLE, sendbuf, pack_size, &position, MPI_COMM_WORLD);
            MPI_Pack(&d[(jc+k)*2], 2, MPI_DOUBLE, sendbuf, pack_size, &position, MPI_COMM_WORLD);
        }
    }

    // 4. 全交换通信
    MPI_Alltoallv(sendbuf, send_counts, sdispls, MPI_PACKED,
                 recvbuf, recv_counts, rdispls, MPI_PACKED,
                 MPI_COMM_WORLD);

    // 5. 使用MPI_Unpack解包数据
    position = 0;
    for (int p = 0; p < size; ++p) {
        const int p_start = p * chunk;
        for (int j = p_start; j < p_start + chunk; ++j) {
            const int jc = j * mj2;
            for (int k = 0; k < mj; ++k) {
                MPI_Unpack(recvbuf, size*pack_size, &position, &c[(jc+k)*2], 2, MPI_DOUBLE, MPI_COMM_WORLD);
                MPI_Unpack(recvbuf, size*pack_size, &position, &d[(jc+k)*2], 2, MPI_DOUBLE, MPI_COMM_WORLD);
            }
        }
    }

    delete[] sendbuf;
    delete[] recvbuf;
}
void step ( int n, int mj, double a[], double b[], double c[],
    double d[], double w[], double sgn )
{
    double ambr;
    double ambu;
    int j;
    int ja;
    int jb;
    int jc;
    int jd;
    int jw;
    int k;
    int lj;
    int mj2;
    double wjw[2];

    mj2 = 2 * mj;
    lj = n / mj2;

    for ( j = 0; j < lj; j++ )
    {
        jw = j * mj;
        ja  = jw;
        jb  = ja;
        jc  = j * mj2;
        jd  = jc;

        wjw[0] = w[jw*2+0]; 
        wjw[1] = w[jw*2+1];

        if ( sgn < 0.0 ) 
        {
            wjw[1] = - wjw[1];
        }

        for ( k = 0; k < mj; k++ )
        {
            c[(jc+k)*2+0] = a[(ja+k)*2+0] + b[(jb+k)*2+0];
            c[(jc+k)*2+1] = a[(ja+k)*2+1] + b[(jb+k)*2+1];

            ambr = a[(ja+k)*2+0] - b[(jb+k)*2+0];
            ambu = a[(ja+k)*2+1] - b[(jb+k)*2+1];

            d[(jd+k)*2+0] = wjw[0] * ambr - wjw[1] * ambu;
            d[(jd+k)*2+1] = wjw[1] * ambr + wjw[0] * ambu;
        }
    }
    return;
}
//****************************************************************************80

void timestamp ( )
{
#define TIME_SIZE 40

    static char time_buffer[TIME_SIZE];
    const struct tm *tm;
    time_t now;

    now = time ( NULL );
    tm = localtime ( &now );

    strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

    std::cout << time_buffer << "\n";

    return;
#undef TIME_SIZE
}