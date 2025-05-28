#include<pthread.h>
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<math.h>
int total;
int in_cycle_sum = 0;
int num_threads;
const long double REAL_PI = 3.14159265358979323846L;

// 结果写入文件
void write_results(int total_points, double para_time, double serial_time, 
                  long double para_pi, long double serial_pi) {
    FILE *fp = fopen("pi_results.txt", "a");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    // 计算误差和效率
    long double para_error = fabsl(para_pi - REAL_PI);
    long double serial_error = fabsl(serial_pi - REAL_PI);
    double efficiency = serial_time / para_time;

    // 写入文件
    fprintf(fp, "%d,",total);
    fprintf(fp, "%d,",num_threads);
    fprintf(fp, "%.2f,",efficiency);
    fprintf(fp, "%.15Lf\n",para_error);
    // fprintf(fp, "Total Points: %d\n", total_points);
    // fprintf(fp, "Parallel Pi: %.15Lf (Error: %.15Lf)\n", para_pi, para_error);
    // fprintf(fp, "Serial Pi:   %.15Lf (Error: %.15Lf)\n", serial_pi, serial_error);
    // fprintf(fp, "Parallel Time: %.6f sec\n", para_time);
    // fprintf(fp, "Serial Time:   %.6f sec\n", serial_time);
    // fprintf(fp, "Efficiency (Speedup): %.2f\n\n", efficiency);

    fclose(fp);
}

void* loacl(void* arg){
    long double x, y;
    int * local_total = (int*) arg;
    int *local_cycle_sum = malloc(sizeof(int));
    int tmp=0;
    for (int i = 0; i < *local_total; i++)
    {
        x = (long double)rand() / RAND_MAX;
        y = (long double)rand() / RAND_MAX;
        
        tmp += (x*x + y*y) <= 1 ? 1 : 0;
    }
    *local_cycle_sum = tmp;
    // printf("%d\n",*local_cycle_sum);
    pthread_exit(local_cycle_sum);
}
int serial(){
    long double x, y;
    // int * local_total = (int*) arg;
    // int *local_cycle_sum = malloc(sizeof(int));
    int tmp=0;
    for (int i = 0; i < total; i++)
    {
        x = (long double)rand() / RAND_MAX;
        y = (long double)rand() / RAND_MAX;
        
        tmp += (x*x + y*y) <= 1 ? 1 : 0;
    }
    // *local_cycle_sum = tmp;
    // printf("%d\n",*local_cycle_sum);
    // pthread_exit(local_cycle_sum);
    return tmp;
}
int main(int argc, char *argv[]){
     if (argc < 3)
     {
          printf("process with ./xx <num_thread><points>");
          return 2;
     }
     
    num_threads = atoi(argv[1]);
    total = atoi(argv[2]);
    pthread_t threads[num_threads];
    int base_total = total / num_threads;
    int rem = total % num_threads;
    int local_totals[num_threads];
    for (int i = 0; i < num_threads; i++)
    {
        local_totals[i] = base_total + ((i < rem) ? 1 : 0);
    }
    struct timeval para_st, para_ed;
    gettimeofday(&para_st, NULL);
    for (int i = 0; i < num_threads; i++)
    {
        pthread_create(&threads[i], NULL, loacl, &local_totals[i]);
    }
    for (int i = 0; i < num_threads; i++)
    {
        int * t;
        pthread_join(threads[i], (void**)&t);
        in_cycle_sum += *t;
        // printf("%d\n", *t);
        free(t);
    }
    long double ans = (long double)in_cycle_sum / (long double)total;
    ans *= 4;
    gettimeofday(&para_ed, NULL);
    double para_time = (para_ed.tv_sec - para_st.tv_sec) + (para_ed.tv_usec - para_st.tv_usec) / 1e6;
    printf("%f\n",para_time);
    printf("%Lf\n", ans);
    struct timeval serial_st, serial_ed;
    gettimeofday(&serial_st, NULL);
    int t=serial();
    long double serial_ans = (long double)t / (long double)total;
    serial_ans *= 4;
    gettimeofday(&serial_ed, NULL);
    double serial_time = (serial_ed.tv_sec - serial_st.tv_sec) + (serial_ed.tv_usec - serial_st.tv_usec) / 1e6;
    printf("%f\n",serial_time);
    
    printf("%Lf", serial_ans);

    write_results(total, para_time, serial_time, ans, serial_ans);
}