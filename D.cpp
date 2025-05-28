#include <iostream>
#include <vector>
#include <cstdlib>
#include <pthread.h>

using namespace std;

vector<int> A;
vector<int> B;
vector<int> C;
vector<int> D;
int N = 256;
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
int st = 0;
int cnt = 16;

void init_matrix(vector<int> &m) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m[i*N+j] = rand() % 100;
        }
    }
}

bool compare(vector<int> &A, vector<int> &B) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (A[i*N+j] != B[i*N+j]) return false;
        }
    }
    return true;
}

void serialMultiply() {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                D[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
}

void* matrixMultiply(void* args) {
    while (true) {
        int start;
        int end;
        pthread_mutex_lock(&mtx);
        if (st >= N) {
            pthread_mutex_unlock(&mtx);
            return nullptr;
        }
        start = st;
        st += cnt;
        end = min(st, N);
        cout << "Thread " << pthread_self() % 100 << " start: " << start << " end: " << end << endl;
        pthread_mutex_unlock(&mtx);
        for (int i = start; i < end; i++) {
            for (int k = 0; k < N; k++) {
                for (int j = 0; j < N; j++) {
                    C[i*N+j] += A[i*N+k] * B[k*N+j];
                }
            }
        }
    }
    return nullptr;
}

int main()
{
    int thread_num = 4;
    A.resize(N*N);
    B.resize(N*N);
    C.resize(N*N);
    D.resize(N*N);
    init_matrix(A);
    init_matrix(B);

    serialMultiply();

    pthread_t threads[thread_num];
    for (int i = 0; i < thread_num; i++) {
        pthread_create(&threads[i], NULL, matrixMultiply, NULL);
    }
    for (int i = 0; i < thread_num; i++) {
        pthread_join(threads[i], NULL);
    }

    if (compare(C, D)) {
        cout << "The result is correct!" << endl;
    }
    else {
        cout << "The result is wrong!" << endl;
    }
    
}