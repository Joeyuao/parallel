#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>

double a, b, c, b_2, ac_4;
double delt;
double x1,x2;

pthread_mutex_t mutex0 = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond0 = PTHREAD_COND_INITIALIZER;
int cnt = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
bool condition = 0;
bool is = false;

void* calculate_b_2(void* arg){
    b_2 = b * b;
    //计算完结果之后将cnt+1
    pthread_mutex_lock(&mutex0);
    cnt++;
    pthread_mutex_unlock(&mutex0);
    //thread 0和1都计算好了，就唤醒thread 2。
    if(cnt == 2)pthread_cond_broadcast(&cond0);
    return NULL;
}

void* calculate_ac_4(void* arg){
    ac_4 = 4 * a *c;
    //计算完结果之后将cnt+1
    pthread_mutex_lock(&mutex0);
    cnt++;
    pthread_mutex_unlock(&mutex0);
    //thread 0和1都计算好了，就唤醒thread 2。
    if(cnt == 2)pthread_cond_broadcast(&cond0);
    return NULL;
}

void* calculate_delt(void* arg){
    //等待条件满足（thread 0，1的中间结果计算完毕）
    // pthread_mutex_lock(&mutex0);
    // while (cnt != 2)
    // {
    //     pthread_cond_wait(&cond0, &mutex0);
    // }
    // pthread_mutex_unlock(&mutex0);
    // //计算delt
    // delt=b_2 - ac_4;
    delt =b * b - 4 * a * c; 
    //唤醒线程3和4
    pthread_mutex_lock(&mutex);
    condition = true;
    pthread_mutex_unlock(&mutex);
    pthread_cond_broadcast(&cond);
    return NULL;
}

void* calculate_x1(void* arg){
    pthread_mutex_lock(&mutex);
    while (condition == false)
    {
        pthread_cond_wait(&cond, &mutex);
    }
    pthread_mutex_unlock(&mutex);
    if(delt < 0){
        printf("has no solution.\n");
        pthread_exit(NULL);
    }
    is = 1;
    x1=(-b + sqrt(delt)) / 2*a;
    return NULL;
}

void* calculate_x2(void* arg){
    pthread_mutex_lock(&mutex);
    while (condition == false)
    {
        pthread_cond_wait(&cond, &mutex);
    }
    pthread_mutex_unlock(&mutex);
    if(delt < 0){
        printf("has no solution.\n");
        pthread_exit(NULL);
    }
    is = 1;
    x2=(-b - sqrt(delt)) / 2*a;
    return NULL;
}
double serial_x1(){
    return ((-b) + sqrt(b * b - 4 * a * c))/(2 * a);
}
double serial_x2(){
    return ((-b) - sqrt(b * b - 4 * a * c))/(2 * a);
}
int main(){
    a = 1.0; b = 3.0; c = 2.0;
    struct timeval para_st, para_ed;
    gettimeofday(&para_st, NULL);
    pthread_t t1, t2, t3, t4, t5;
    // pthread_create(&t1, NULL, calculate_b_2, NULL);
    // pthread_create(&t2, NULL, calculate_ac_4, NULL);
    pthread_create(&t3, NULL, calculate_delt, NULL);
    pthread_create(&t4, NULL, calculate_x1, NULL);
    pthread_create(&t5, NULL, calculate_x2, NULL);
    

    // pthread_join(t1, NULL);
    // pthread_join(t2, NULL);
    pthread_join(t3, NULL);
    pthread_join(t4, NULL);
    pthread_join(t5, NULL);
    gettimeofday(&para_ed, NULL);
    double para_time = (para_ed.tv_sec - para_st.tv_sec) + (para_ed.tv_usec - para_st.tv_usec) / 1e6;
    printf("parallel time cost:%f\n", para_time);
    // printf("%f,%f",b_2,ac_4);
    if(is)printf("x1:%lf ;x2:%lf\n", x1, x2);
    
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
    pthread_mutex_destroy(&mutex0);
    pthread_cond_destroy(&cond0);
    
    double x11, x22;
    struct timeval serial_st, serial_ed;
    gettimeofday(&serial_st, NULL);
    x11 = serial_x1();
    x22 = serial_x2();
    gettimeofday(&serial_ed, NULL);
    double serial_time = (serial_ed.tv_sec - serial_st.tv_sec) + (serial_ed.tv_usec - serial_st.tv_usec) / 1e6;
    printf("serial time cost:%f\n", serial_time);
    if(is)printf("x1:%lf ;x2:%lf\n", x11, x22);
    return 0;
}