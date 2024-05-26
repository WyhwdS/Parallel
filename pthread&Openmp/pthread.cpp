#include <iostream>
#include <cmath>
#include <pthread.h>
#include <atomic>
#include<stdlib.h>
#include<chrono>
#include<sys/time.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<tmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>

#define THREAD_NUM 16 // 使用 16 个线程
#define maxn 500 // 矩阵大小

using namespace std;
using namespace chrono;

float** matrix = (float**)_aligned_malloc(maxn*sizeof(float*), 16);
std::atomic<int> next_row(0);

void m_set(float **m){
    for(int i=0;i<maxn;i++){
        int j;
        for(int j=0;j<maxn;j++){
            if(i==j){
                m[i][j]=1.0;
            }
            else
                m[i][j]=rand()%1000;
        }
    }
    for(int k=0;k<maxn;k++){
        for(int i=k+1;i<maxn;i++){
            for(int j=0;j<maxn;j++){
                m[i][j]+=m[k][j];
                m[i][j]=(int)m[i][j]%1000;
            }
        }
    }
}

void normal_GE(float **m){
    for(int k=0;k<maxn;k++){
        for(int j=k+1;j<maxn;j++){
            m[k][j]=m[k][j]/m[k][k];
        }
        m[k][k]=1.0;
        for(int i=k+1;i<maxn;i++){
            for(int j=k+1;j<maxn;j++){
                m[i][j]=m[i][j]-m[i][k]*m[k][j];
            }
            m[i][k]=0;
        }
    }
}

void *thread_func(void *arg) {
    while (true) {
        int row = next_row.fetch_add(1); // 动态获取下一个行
        if (row >= maxn) break; // 所有行都处理完了

        for (int j = row + 1; j < maxn; j++) {
            matrix[row][j] = matrix[row][j] / matrix[row][row];
        }
        matrix[row][row] = 1.0;
        for (int i = row + 1; i < maxn; i++) {
            for (int j = row + 1; j < maxn; j++) {
                matrix[i][j] = matrix[i][j] - matrix[i][row] * matrix[row][j];
            }
            matrix[i][row] = 0;
        }
    }
    pthread_exit(NULL);
    return nullptr;
}

void pthread() {
    pthread_t threads[THREAD_NUM];

    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_create(&threads[i], NULL, thread_func, NULL);
    }

    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_join(threads[i], NULL);
    }

    next_row.store(0, std::memory_order_relaxed);
}


void *thread_func_SSE(void *arg) {
    while (true) {
        int row = next_row.fetch_add(1); // 动态获取下一个行
        if (row >= maxn) break; // 所有行都处理完了

        int j = 0;
        __m128 vt = _mm_set1_ps(matrix[row][row]);
        for (j = row + 1; j + 4 <= maxn; j += 4) {                
            __m128 va = _mm_loadu_ps(&matrix[row][j]);	
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(&matrix[row][j], va);
        }
        for (j; j < maxn; j++) {
            matrix[row][j] = matrix[row][j] / matrix[row][row];
        }

        matrix[row][row] = 1.0;
        for (int i = row + 1; i < maxn; i++) {
            __m128 vaik =_mm_set1_ps(matrix[i][row]);                
            for (j = row + 1; j + 4 <= maxn; j += 4) {
                __m128 vakj = _mm_loadu_ps(&matrix[row][j]);
                __m128 vaij = _mm_loadu_ps(&matrix[i][j]);
                __m128 vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&matrix[i][j], vaij);
            }
            for (j ; j < maxn; j++) {
                matrix[i][j] = matrix[i][j] - matrix[row][j] * matrix[i][row];
            }
            matrix[i][row] = 0;
        }
    }
    pthread_exit(NULL);
    return nullptr;
}

void pthread_SSE() {
    pthread_t threads[THREAD_NUM];

    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_create(&threads[i], NULL, thread_func_SSE, NULL);
    }

    for (int i = 0; i < THREAD_NUM; i++) {
        pthread_join(threads[i], NULL);
    }

    next_row.store(0, std::memory_order_relaxed);
}



int main() {
    // 初始化矩阵
    for (int i = 0; i < maxn; i++) {
		matrix[i] = (float*)_aligned_malloc(maxn*sizeof(float), 16);
	}
    // 执行并行 Gauss-Jordan 消元
    
    m_set(matrix);
    auto start = high_resolution_clock::now();
    normal_GE(matrix);
    auto finish = high_resolution_clock::now();
    duration<double> duration = finish -start;
    cout<<"normal:"<<1000*duration.count()<<"ms\n";

    m_set(matrix);
    start = high_resolution_clock::now();
    pthread();
    finish = high_resolution_clock::now();
    duration = finish -start;
    cout<<"pthread:"<<1000*duration.count()<<"ms\n";

    m_set(matrix);
    start = high_resolution_clock::now();
    pthread_SSE();
    finish = high_resolution_clock::now();
    duration = finish -start;
    cout<<"pthread&SSE:"<<1000*duration.count()<<"ms\n";

    system("pause");
    return 0;
}