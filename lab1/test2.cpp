#include<iostream>
#include<sys/time.h>
using namespace std;
#define n 100000000
int num[2*n];
long long sum1,sum2;
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
        num[i]=i;
    }
    double time_sum=0.0;
    double start,finish;
    int round=10;
    for(int r=0;r<round;r++)
    {
        start=time();
        for(int i=0;i<n;i++)
        {
            sum1+=num[i];
        }
        finish=time();
        time_sum+=(finish-start);
    }
    cout<<"normal:"<<time_sum/1000/round<<"ms\n";

    time_sum=0.0;
    for(int r=0;r<round;r++)
    {
        start = time();
        for(int m=n;m>1;m/=2)
        {
            for(int i=0;i<m;i++)
            {
                num[i]=num[i*2]+num[i*2+1];
            }
        }
        finish = time();
        time_sum+=(finish-start);
    }
    cout<<"better:"<<time_sum/1000/round<<"ms\n";
}