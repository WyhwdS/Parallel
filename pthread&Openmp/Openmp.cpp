#include <iostream>
#include <cmath>
#include <omp.h>
#include <stdlib.h>
#include <chrono>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<tmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>

#define maxn 500 // 矩阵大小

using namespace std;
using namespace chrono;

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

void openmp(float** m) {
    omp_set_num_threads(16);
    #pragma omp parallel for
    for (int k = 0; k < maxn; k++) {
        for (int j = k + 1; j < maxn; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < maxn; i++) {
            for (int j = k + 1; j < maxn; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void openmp_SSE(float** m) {
    omp_set_num_threads(16);
    #pragma omp parallel for
    for (int k = 0; k < maxn; k++) {
        int j = 0;
        __m128 vt = _mm_set1_ps(m[k][k]);
        for (j = k + 1; j + 4 <= maxn; j += 4) {
            __m128 va = _mm_loadu_ps(&m[k][j]);	
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(&m[k][j], va);
        }
        for (j; j < maxn; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < maxn; i++) {
            __m128 vaik =_mm_set1_ps(m[i][k]);
            for (j = k + 1; j + 4 <= maxn; j += 4) {
                __m128 vakj = _mm_loadu_ps(&m[k][j]);
                __m128 vaij = _mm_loadu_ps(&m[i][j]);
                __m128 vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(&m[i][j], vaij);
            }
            for (j ; j < maxn; j++) {
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            }
            m[i][k] = 0;
        }    
    }
}

int main() {
    float** matrix = (float**)_aligned_malloc(maxn*sizeof(float*), 16);
    for (int i = 0; i < maxn; i++) {
		matrix[i] = (float*)_aligned_malloc(maxn*sizeof(float), 16);
	}
    m_set(matrix);

    // 执行并行 Gauss-Jordan 消元
    auto start = high_resolution_clock::now();
    normal_GE(matrix);
    auto finish = high_resolution_clock::now();
    duration<double> duration = finish - start;
    cout << "normal:" << 1000 * duration.count() << "ms\n";


    m_set(matrix);
    start = high_resolution_clock::now();
    openmp(matrix);
    finish = high_resolution_clock::now();
    duration = finish -start;
    cout<<"openmp:"<<1000*duration.count()<<"ms\n";

    m_set(matrix);
    start = high_resolution_clock::now();
    openmp_SSE(matrix);
    finish = high_resolution_clock::now();
    duration = finish -start;
    cout<<"openmp&SSE:"<<1000*duration.count()<<"ms\n";

    system("pause");
    return 0;
}