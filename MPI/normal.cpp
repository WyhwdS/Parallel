#include <iostream>
#include <cmath>
#include <omp.h>
#include <stdlib.h>
#include<sys/time.h>
#include<mpi.h>
#define maxn 1800 // 矩阵大小

using namespace std;
float matrix[maxn][maxn];
double time()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    double time = tv.tv_sec*1000000+tv.tv_usec;
    return time;
}

void m_set(){
    for(int i=0;i<maxn;i++){
        for(int j=0;j<maxn;j++){
            if(i==j){
                matrix[i][j]=1.0;
            }
            else
                matrix[i][j]=rand()%1000;
        }
    }
    for(int k=0;k<maxn;k++){
        for(int i=k+1;i<maxn;i++){
            for(int j=0;j<maxn;j++){
                matrix[i][j]+=matrix[k][j];
                matrix[i][j]=(int)matrix[i][j]%1000;
            }
        }
    }
}

void normal_GE(){
    for(int k=0;k<maxn;k++){
        for(int j=k+1;j<maxn;j++){
            matrix[k][j]=matrix[k][j]/matrix[k][k];
        }
        matrix[k][k]=1.0;
        for(int i=k+1;i<maxn;i++){
            for(int j=k+1;j<maxn;j++){
                matrix[i][j]=matrix[i][j]-matrix[i][k]*matrix[k][j];
            }
            matrix[i][k]=0;
        }
    }
}
int main(){
    
    m_set();

    double start,finish;

    start=time();
    normal_GE();
    finish = time();
    cout<<"normal:"<<(finish-start)/1000<<"ms\n";
}