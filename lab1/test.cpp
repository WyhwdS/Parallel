#include<iostream>
#include<sys/time.h>
using namespace std;
#define n 12000
long long a[n],b[n][n],sum[n];
double time()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    double time = tv.tv_sec*1000000+tv.tv_usec;
    return time;
}
int main()
{
    for(int i=0;i<n;i++)
    {
        a[i]=i;
    }
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            b[i][j]=i+j;
        }
    }
    double time_sum=0.0;
    double start,finish;
    int round=10;
    for(int r=0;r<round;r++)
    {
        start=time();
        for(int i=0;i<n;i++)
        {
            sum[i]=0;
            for(int j=0;j<n;j++)
            {
                sum[i]+=b[j][i]*a[j];
            }
        }
        finish = time();
        time_sum+=(finish-start);
    }
    
    cout<<"normal:"<<time_sum/1000/round<<"ms\n";

    time_sum=0.0;
    for(int r=0;r<round;r++)
    {
        start = time();
        for(int i=0;i<n;i++)
        {
            sum[i]=0.0;
        }
        for(int j=0;j<n;j++)
        {
            for(int i=0;i<n;i++)
            {
                sum[i]+=b[j][i]*a[j];
            }
        }
        finish = time();
        time_sum+=(finish-start);
    }
    cout<<"better:"<<time_sum/1000/round<<"ms\n";
}