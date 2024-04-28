#include<iostream>
#include<stdlib.h>
#include<chrono>
#include<sys/time.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<tmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
using namespace std;
using namespace chrono;
#define maxn 512
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

void SSE(float** m) {
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


void only_div_SSE(float** m) {
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

        for(int i=k+1;i<maxn;i++)
        {
            for(int j=k+1;j<maxn;j++)
            {
                m[i][j]=m[i][j]-m[i][k]*m[k][j];
            }
            m[i][k]=0;
        }
	}
}

void only_sub_SSE(float** m) {
	for (int k = 0; k < maxn; k++) {
        int j = 0;
        for(int j=k+1;j<maxn;j++)
        {
            m[k][j]=m[k][j]/m[k][k];
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
    float** A = (float**)_aligned_malloc(maxn*sizeof(float*), 16);
    for (int i = 0; i < maxn; i++) {
		A[i] = (float*)_aligned_malloc(maxn*sizeof(float), 16);
	}
    m_set(A);
    auto start = high_resolution_clock::now();
    normal_GE(A);
    auto finish = high_resolution_clock::now();
    duration<double> duration = finish -start;
    cout<<"normal:"<<1000*duration.count()<<"ms\n";

    m_set(A);
    start = high_resolution_clock::now();
    SSE(A);
    finish = high_resolution_clock::now();
    duration = finish -start;
    cout<<"SSE:"<<1000*duration.count()<<"ms\n";

    m_set(A);
    start = high_resolution_clock::now();
    only_sub_SSE(A);
    finish = high_resolution_clock::now();
    duration = finish -start;
    cout<<"only_sub_SSE:"<<1000*duration.count()<<"ms\n";

    m_set(A);
    start = high_resolution_clock::now();
    only_div_SSE(A);
    finish = high_resolution_clock::now();
    duration = finish -start;
    cout<<"only_div_SSE:"<<1000*duration.count()<<"ms\n";
    
    
    // for(int i=0;i<maxn;i++)
    // {
    //     for(int j=0;j<maxn;j++)
    //     {
    //         cout<<A[i][j]<< " ";
    //     }
    //     cout<<endl;
    // }
    return 0;
}