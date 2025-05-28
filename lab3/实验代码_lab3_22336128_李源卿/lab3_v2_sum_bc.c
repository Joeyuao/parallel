#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <sys/time.h>
int BLOCK_SIZE;

int array_size;
int start=0;
int end;

typedef struct {
    int *array;
    long long *partial_sum;
} ThreadArgs;
pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
void* compute_sum(void *arg) {
    while (1)
    {
        int st, ed;
        pthread_mutex_lock(&mtx);
        if(start >= array_size){
            pthread_mutex_unlock(&mtx);
            pthread_exit(NULL);
        }
        st = start;
        ed = end < array_size ? end : array_size;
        start += BLOCK_SIZE;
        end += BLOCK_SIZE;
        // end = 
        pthread_mutex_unlock(&mtx);
        ThreadArgs *args = (ThreadArgs*) arg;
        long long sum = 0;
        for (int i = st; i < ed; ++i) {
            sum += args->array[i];
        }
        *(args->partial_sum) = *(args->partial_sum) + sum;
    }
    
    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <num_threads> <array_size> <BLOCK_PERCENT>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    array_size = atoi(argv[2]);
    int BLOCK_PERCENT = atoi(argv[3]);
    //注意不要吧BLOCK_SIZE 除成0
    BLOCK_SIZE = array_size / num_threads / BLOCK_PERCENT;
    if(BLOCK_SIZE == 0){
        printf("please check your input!\n");
        return 2;
    }
    end = BLOCK_SIZE;
    // 分配并初始化数组
    int *array = malloc(array_size * sizeof(int));
    for (int i = 0; i < array_size; ++i) {
        array[i] = 1; // 全1数组方便验证结果
    }

    // 分配部分和数组
    long long *partial_sums = calloc(num_threads, sizeof(long long));

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    ThreadArgs *thread_args = malloc(num_threads * sizeof(ThreadArgs));

    int block_size = array_size / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        thread_args[i].array = array;
        // thread_args[i].start = i * block_size;
        // thread_args[i].end = (i == num_threads - 1) ? array_size - 1 : (i + 1) * block_size - 1;
        thread_args[i].partial_sum = &partial_sums[i];
    }

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // 创建线程
    for (int i = 0; i < num_threads; ++i) {
        pthread_create(&threads[i], NULL, compute_sum, &thread_args[i]);
    }

    // 等待线程结束
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

    // 汇总结果
    long long total_sum = 0;
    for (int i = 0; i < num_threads; ++i) {
        total_sum += partial_sums[i];
    }

    gettimeofday(&end, NULL);
    double p_time = (end.tv_sec - start.tv_sec) ;
    // printf("%lf\n",p_time);
    p_time = p_time + (end.tv_usec - start.tv_usec) * 1e-6;

    // gettimeofday(&start, NULL);
    // long long s_sum = 0;
    // for (int i = 0; i < array_size; i++)
    // {
    //     s_sum += array[i];
    // }
    // gettimeofday(&end, NULL);
    // double s_time = (end.tv_sec - start.tv_sec) ;
    // s_time = s_time + (end.tv_usec - start.tv_usec) * 1e-6;
    // printf("Total sum: %lld\n", total_sum);
    // printf("is correct :%s \n", s_sum == total_sum ? "YES" : "NO");
    // printf("serial time taken: %.8f seconds\n", s_time);
    printf("parallel time taken: %.8f seconds\n", p_time);
    // printf("Speedup: %.2f\n", s_time / p_time);
    // printf("Efficiency: %.2f%%\n", (s_time / p_time) / num_threads * 100);
    // 清理资源
    free(array);
    free(partial_sums);
    free(threads);
    free(thread_args);

    return 0;
}