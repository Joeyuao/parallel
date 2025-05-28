#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>

// 线程参数结构体
struct ThreadArgs{
    long long iterations;        // 本线程需要计算的迭代次数
    struct drand48_data rand_state;  // 线程独立的随机数状态
} __attribute__((aligned(64)));

// 全局变量
long long total;
long long in_cycle_sum = 0;
long long num_threads;
const long double REAL_PI = 3.14159265358979323846L;

// 结果写入文件
void write_results(long long total_points, double para_time, double serial_time, 
                  long double para_pi, long double serial_pi) {
    FILE *fp = fopen("pi_results1.txt", "a");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    long double para_error = fabsl(para_pi - REAL_PI);
    double efficiency = serial_time / para_time / num_threads;

    fprintf(fp, "%lld,%lld,%.2f,%.15Lf,%f,%f,%.2f\n", 
            total_points, num_threads, efficiency, para_error,serial_time,para_time,serial_time / para_time);
    fclose(fp);
}

// 线程函数
void* local(void* arg) {
    struct ThreadArgs *args = (struct ThreadArgs*) arg;
    long long *local_sum = malloc(sizeof(long long));
    *local_sum = 0;

    for (long long i = 0; i < args->iterations; i++) {
        double x, y;
        drand48_r(&args->rand_state, &x);  // 生成[0,1)的随机数
        drand48_r(&args->rand_state, &y);
        *local_sum += (x*x + y*y) <= 1.0;
    }

    pthread_exit(local_sum);
}

// 串行版本
long long serial() {
    struct drand48_data rand_state;
    srand48_r(time(NULL), &rand_state);  // 初始化随机数状态

    long long sum = 0;
    for (long long i = 0; i < total; i++) {
        double x, y;
        drand48_r(&rand_state, &x);
        drand48_r(&rand_state, &y);
        sum += (x*x + y*y) <= 1.0;
    }
    return sum;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num_threads> <total_points>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // 解析参数
    num_threads = atoi(argv[1]);
    total = atoi(argv[2]);
    
    if (num_threads <= 0 || total <= 0) {
        fprintf(stderr, "Arguments must be positive integers\n");
        return EXIT_FAILURE;
    }

    // === 并行计算 ===
    pthread_t threads[num_threads];
    struct ThreadArgs args[num_threads];
    
    // 分配任务并初始化随机数状态
    long long base_total = total / num_threads;
    long long rem = total % num_threads;
    for (long long i = 0; i < num_threads; i++) {
        args[i].iterations = base_total + (i < rem ? 1 : 0);
        // 生成唯一种子：时间戳 + 线程索引
        long seed = time(NULL) ^ (i + 1);
        srand48_r(seed, &args[i].rand_state);
    }

    struct timeval para_st, para_ed;
    gettimeofday(&para_st, NULL);

    // 创建线程
    for (long long i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, local, &args[i]);
    }

    // 回收结果
    in_cycle_sum = 0;
    for (long long i = 0; i < num_threads; i++) {
        long long *local_sum;
        pthread_join(threads[i], (void**)&local_sum) != 0;
        in_cycle_sum += *local_sum;
        free(local_sum);
    }

    // 计算结果
    long double para_pi = 4.0L * in_cycle_sum / total;
    gettimeofday(&para_ed, NULL);
    double para_time = (para_ed.tv_sec - para_st.tv_sec) + 
                      (para_ed.tv_usec - para_st.tv_usec) / 1e6;

    // === 串行计算 ===
    struct timeval serial_st, serial_ed;
    gettimeofday(&serial_st, NULL);
    long long serial_sum = serial();
    long double serial_pi = 4.0L * serial_sum / total;
    gettimeofday(&serial_ed, NULL);
    double serial_time = (serial_ed.tv_sec - serial_st.tv_sec) + 
                        (serial_ed.tv_usec - serial_st.tv_usec) / 1e6;

    // 输出结果
    printf("Parallel Time: %.6fs\n", para_time);
    printf("Parallel Pi: %.15Lf\n", para_pi);
    printf("Serial Time: %.6fs\n", serial_time);
    printf("Serial Pi: %.15Lf\n", serial_pi);

    // 写入结果文件
    write_results(total, para_time, serial_time, para_pi, serial_pi);

    return EXIT_SUCCESS;
}