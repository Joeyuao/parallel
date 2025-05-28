#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include "parallel_for.h"
#include <sys/time.h>
#define M 500
#define N 500
void print_mat(double mat[M][N]){
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      printf("%lf ",mat[i][j]);
    }
    printf("\n");
  }
  
}
typedef struct 
{
    double mean;
    double (*w)[N];
}InitArgs;

// 用于求和操作的结构体
typedef struct {
    double (*w)[N];      // 二维数组指针
    double *local_sums;  // 各线程的局部和数组
    int thread_count;    // 线程总数

} SumData;

// 用于拷贝操作的结构体
typedef struct {
    double (*u)[N];
    double (*w)[N];
} CopyArgs;

// 用于计算新温度的结构体
typedef struct {
    double (*u)[N];
    double (*w)[N];
} ComputeArgs;

// 用于计算最大温差的结构体
typedef struct {
    double (*u)[N];
    double (*w)[N];
    double *local_diffs; // 各线程的局部最大值
    int thread_count;
} DiffData;

// 设置左边界（i循环）
void set_left_boundary(int i, int tid,void *arg) {
    double (*w)[N] = arg;
    w[i][0] = 100.0;
}

// 设置右边界（i循环）
void set_right_boundary(int i, int tid,void *arg) {
    double (*w)[N] = arg;
    w[i][N-1] = 100.0;
}

// 设置底部边界（j循环）
void set_bottom_boundary(int j, int tid,void *arg) {
    double (*w)[N] = arg;
    w[M-1][j] = 100.0;
}

// 设置顶部边界（j循环）
void set_top_boundary(int j, int tid,void *arg) {
    double (*w)[N] = arg;
    w[0][j] = 0.0;
}

// 计算i方向的边界和（局部和版本）
void sum_i_contribution(int i, int tid,void *arg) {
    SumData *data = (SumData *)arg;
    // 通过线程参数位置计算线程ID
    data->local_sums[tid] += data->w[i][0] + data->w[i][N-1];
}

// 计算j方向的边界和（局部和版本）
void sum_j_contribution(int j, int tid,void *arg) {
    SumData *data = (SumData *)arg;
    data->local_sums[tid] += data->w[M-1][j] + data->w[0][j];
}

// 初始化内部温度
void init_internal(int i, int tid,void *arg) {
    InitArgs* args = (InitArgs*) arg;
    // printf("mean:%lf",mean);
    for (int j = 1; j < N-1; j++) {
        args->w[i][j] = args->mean;
    }
}

// 拷贝温度矩阵
void copy_w_to_u(int i, int tid,void *arg) {
    CopyArgs *args = (CopyArgs *)arg;
    for (int j = 0; j < N; j++) {
        args->u[i][j] = args->w[i][j];
    }
}

// 计算新的温度值
void compute_new_w(int i, int tid,void *arg) {
    ComputeArgs *args = (ComputeArgs *)arg;
    for (int j = 1; j < N-1; j++) {
        args->w[i][j] = (args->u[i-1][j] + args->u[i+1][j] 
                       + args->u[i][j-1] + args->u[i][j+1]) / 4.0;
    }
}
// 计算当前迭代的最大温差（局部最大值版本）
void compute_diff(int i, int tid,void *arg) {
    // printf("line:%d tid:%d ",i,tid);
    DiffData *data = (DiffData *)arg;
    double local_max = 0.0;
    for (int j = 1; j < N-1; j++) {
        double current_diff = fabs(data->w[i][j] - data->u[i][j]);
        if (current_diff > local_max) {
            local_max = current_diff;
        }
    }
    // 更新为当前最大值与线程局部最大值的较大者
    if (local_max > data->local_diffs[tid]) {
        // printf("local_max :%lf ",data->local_diffs[tid]);
        data->local_diffs[tid] = local_max;
        // printf("local_max changes:%lf ",data->local_diffs[tid]);
    }
    //  printf("loacal_max:%lf\n",data->local_diffs[tid]);
}


// 辅助函数：执行并行计算并汇总结果
double parallel_sum(int start, int end, int inc, 
                   void (*functor)(int, int,void*), 
                   int num_threads, SumData *data) {
    // 分配局部和数组（第一个元素存储线程数）
    data->local_sums = calloc(num_threads, sizeof(double));
    data->thread_count = num_threads;
    
    // 执行并行计算
    parallel_for(start, end, inc, functor, data, num_threads);

    // 汇总结果
    double total = 0.0;
    for (int i = 0; i < num_threads; i++) {
        // printf("%lf ",data->local_sums[i]);
        total += data->local_sums[i];
    }
    // printf("\n");
    free(data->local_sums);
    return total;
}

// 辅助函数：计算最大差值
double parallel_max_diff(int num_threads, DiffData *data) {
    // 分配局部最大值数组（第一个元素存储线程数）
    data->local_diffs = calloc(num_threads+1, sizeof(double));
    data->thread_count = num_threads;

    // 执行并行计算
    parallel_for(1, M-1, 1, compute_diff, data, num_threads);

    // 寻找全局最大值
    double global_max = 0.0;
    // printf("diffs:\n");
    for (int i = 0; i < num_threads; i++) {
    
        // printf("%lf ",data->local_diffs[i]);
        if (data->local_diffs[i] > global_max) {
            global_max = data->local_diffs[i];
        }
    }
    // printf("\n");
    free(data->local_diffs);
    return global_max;
}

int main(int argc, char *argv[]) {
    double epsilon = 0.005;
    int iterations = 0;
    int iterations_print = 1;
    double u[M][N];
    double w[M][N];
    double mean;
    double diff = epsilon;
    double wtime;

    // 设置线程数（示例使用4线程，实际可通过参数指定）
    int num_threads;
    if(argc > 1) {
      num_threads = atoi(argv[1]);
    } else {
        num_threads = 4;
    }

    printf("\nHEATED_PLATE_PTHREAD\n  C/Pthreads version\n");
    printf("  Spatial grid of %d by %d points\n", M, N);
    printf("  Using %d threads\n", num_threads);
    printf("  Iteration until change <= %e\n", epsilon);

    /* 设置边界条件 */
    parallel_for(1, M-1, 1, set_left_boundary, w, num_threads);
    parallel_for(1, M-1, 1, set_right_boundary, w, num_threads);
    parallel_for(0, N, 1, set_bottom_boundary, w, num_threads);
    parallel_for(0, N, 1, set_top_boundary, w, num_threads);

    /* 计算边界平均值 */
    SumData sum_data_i = {w};
    SumData sum_data_j = {w};
    double sum_i = parallel_sum(1, M-1, 1, sum_i_contribution, num_threads, &sum_data_i);
    double sum_j = parallel_sum(0, N, 1, sum_j_contribution, num_threads, &sum_data_j);
    
    mean = (sum_i + sum_j) / (2*M + 2*N - 4);
    // printf("mean =  %lf\n",mean);
    printf("\nMEAN = %f\n", mean);

    /* 初始化内部温度 */
    InitArgs init_mean = {mean,w};
    parallel_for(1, M-1, 1, init_internal, &init_mean, num_threads);

    struct timeval st, ed;
    gettimeofday(&st,NULL);
    /* 主迭代循环 */
    // int n = 1024;
    while (epsilon<=diff) {
        // 拷贝温度矩阵
        // print_mat(w);
        CopyArgs copy_args = {u, w};
        parallel_for(0, M, 1, copy_w_to_u, &copy_args, num_threads);
        // print_mat(u);
        // 计算新温度
        ComputeArgs compute_args = {u, w};
        parallel_for(1, M-1, 1, compute_new_w, &compute_args, num_threads);
        // print_mat(w);
        // 计算最大温差
        DiffData diff_data = {u, w};
        diff = parallel_max_diff(num_threads, &diff_data);

        iterations++;
        if (iterations == iterations_print) {
            printf("  %8d  %f\n", iterations, diff);
            iterations_print *= 2;
        }
    }
    gettimeofday(&ed,NULL);
    wtime = ed.tv_sec - st.tv_sec + (ed.tv_usec - st.tv_usec) * 1e-6;
    
    printf("\nFinal iteration: %d  Change: %f\n", iterations, diff);
    printf("Total wallclock time: %.6f seconds\n", wtime);
    printf("\nHEATED_PLATE_PTHREAD: Normal termination\n");
    
    return 0;
}