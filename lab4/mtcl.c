#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <errno.h>

// 线程参数结构体
struct thread_args {
    long long iterations;
    uint32_t seed;
}__attribute__((aligned(64)));

// 全局变量
long long total;
long long in_cycle_sum = 0;
long long num_threads;
const long double REAL_PI = 3.14159265358979323846L;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
// Xorshift32随机数生成器（线程安全）
uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

// 生成[0,1)范围的随机数
double rand_double(uint32_t *state) {
    return (double)xorshift32(state) / (double)UINT32_MAX;
}

// 写入结果到文件
void write_results(long long total_points, double para_time, double serial_time, 
                  long double para_pi, long double serial_pi) {
    FILE *fp = fopen("pi_results.txt", "a");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    long double para_error = fabsl(para_pi - REAL_PI);
    double efficiency = serial_time / para_time;

    fprintf(fp, "%lld,%lld,%.2f,%.15Lf\n", 
            total_points, num_threads, efficiency, para_error);
    fclose(fp);
}

// 线程函数
void* local(void* arg) {
    struct thread_args *args = (struct thread_args*) arg;
    long long local_sum = 0;
    // printf("%lld\n",args->iterations);
    for (long long i = 0; i < args->iterations; i++) {
        double x = rand_double(&args->seed);
        double y = rand_double(&args->seed);
        local_sum += (x*x + y*y) <= 1.0;
    }

    pthread_mutex_lock(&mutex);
    in_cycle_sum += local_sum;
    pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}

// 串行版本
long long serial() {
    uint32_t seed = time(NULL);
    long long sum = 0;

    for (long long i = 0; i < total; i++) {
        double x = rand_double(&seed);
        double y = rand_double(&seed);
        sum += (x*x + y*y) <= 1.0;
    }

    return sum;
}

// 获取当前时间（秒）
double get_current_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num_threads> <total_points>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // 解析参数
    num_threads = atoi(argv[1]);
    total = atoll(argv[2]);
    // printf("%s",argv[2]);
    // printf("%lld",total);
    if (num_threads <= 0 || total <= 0) {
        fprintf(stderr, "Arguments must be positive integers\n");
        return EXIT_FAILURE;
    }

    pthread_t threads[num_threads];
    struct thread_args args[num_threads];

    // 初始化种子
    uint32_t base_seed = time(NULL);
    
    // 分配任务
    long long base_total = total / num_threads;
    long long rem = total % num_threads;
    
    for (long long i = 0; i < num_threads; i++) {
        args[i].iterations = base_total + (i < rem ? 1 : 0);
        args[i].seed = base_seed + i;
    }

    // === 并行计算 ===
    double para_start = get_current_time();
    
    // 创建线程
    for (long long i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, local, &args[i]) != 0) {
            perror("Thread creation failed");
            return EXIT_FAILURE;
        }
    }

    // 回收结果
    in_cycle_sum = 0;
    for (long long i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        // long long *local_sum;
        // if (pthread_join(threads[i], (void**)&local_sum) != 0) {
        //     perror("Thread join failed");
        //     continue;
        // }
        // in_cycle_sum += *local_sum;
        // free(local_sum);
    }

    // 计算结果
    long double para_pi = 4.0L * (long double)in_cycle_sum / (long double)total;
    double para_time = get_current_time() - para_start;

    // === 串行计算 ===
    double serial_start = get_current_time();
    long long serial_sum = serial();
    long double serial_pi = 4.0L * (long double)serial_sum / (long double)total;
    double serial_time = get_current_time() - serial_start;

    // === 输出结果 ===
    printf("=== 蒙特卡洛π计算 ===\n");
    printf("总点数: %lld\n", total);
    printf("线程数: %lld\n", num_threads);
    printf("真实π值: %.15Lf\n", REAL_PI);
    printf("并行π值: %.15Lf (耗时: %.6fs)\n", para_pi, para_time);
    printf("串行π值: %.15Lf (耗时: %.6fs)\n", serial_pi, serial_time);
    printf("加速比: %.2fx\n", serial_time / para_time);
    
    // 写入结果文件
    write_results(total, para_time, serial_time, para_pi, serial_pi);

    return EXIT_SUCCESS;
}