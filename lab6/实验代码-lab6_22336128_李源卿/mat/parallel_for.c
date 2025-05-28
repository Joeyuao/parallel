#include <pthread.h>
#include <stdlib.h>

typedef struct {
    int start_idx;
    int num_iters;
    int inc;
    void *(*functor)(int, void *);
    void *arg;
} ThreadArgs;

static void *thread_func(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    int current = args->start_idx;
    for (int t = 0; t < args->num_iters; t++) {
        args->functor(current, args->arg);
        current += args->inc;
    }
    return NULL;
}

void parallel_for(int start, int end, int inc, void *(*functor)(int, void*), void *arg, int num_threads){
    if (start >= end || inc <= 0 || num_threads <= 0) return;

    // 计算总迭代次数
    int total = ((end - start) + inc - 1) / inc;
    if (total <= 0) return;

    // 限制线程数不超过总迭代次数
    if (num_threads > total) num_threads = total;

    int base = total / num_threads;
    int remainder = total % num_threads;

    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    ThreadArgs *args = malloc(num_threads * sizeof(ThreadArgs));

    for (int k = 0; k < num_threads; k++) {
        int m = base + (k < remainder ? 1 : 0);
        int sum_prev = base * k + (k < remainder ? k : remainder);
        args[k].start_idx = start + sum_prev * inc;
        args[k].num_iters = m;
        args[k].inc = inc;
        args[k].functor = functor;
        args[k].arg = arg;

        pthread_create(&threads[k], NULL, thread_func, &args[k]);
    }

    // 等待所有线程完成
    for (int k = 0; k < num_threads; k++) {
        pthread_join(threads[k], NULL);
    }

    free(threads);
    free(args);
}