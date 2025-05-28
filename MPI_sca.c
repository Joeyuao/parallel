#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
// #include<sys>
// A:m*k B:k*n C:m*n
void matrix_multiply(double *A, double *B, double *C, int rows, int n, int kk) {
    for (int i = 0; i < rows; ++i) {          
        for (int k = 0; k < kk; ++k) {     
            double a = A[i*kk+k];                
            for (int j = 0; j < n; ++j) {  
                C[i*n+j] += a * B[k*n+j];   
            }
        }
    }
}
void print_mat(double* A,int m,int n){
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%lf ",A[i*n+j]);
        }
        printf("\n");
    }
}
typedef struct 
{
    int m;
    int k;
    int n;
}mkn;

int main(int argc, char** argv) {
    
    printf("%d\n",argc);
    for (int i = 0; i < argc; i++)
    {
        printf("%s\n",*(argv+i));
    }
    
    // MPI_Finalize();

    return 0;
}