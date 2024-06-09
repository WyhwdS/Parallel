#include <iostream>
#include <cmath>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>
#include<omp.h>
#define maxn 1800 // 矩阵大小

using namespace std;
float matrix[maxn][maxn];

double time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double time = tv.tv_sec * 1000000 + tv.tv_usec;
    return time;
}

void m_set() {
    for (int i = 0; i < maxn; i++) {
        for (int j = 0; j < maxn; j++) {
            if (i == j) {
                matrix[i][j] = 1.0;
            } else {
                matrix[i][j] = rand() % 1000;
            }
        }
    }
    for (int k = 0; k < maxn; k++) {
        for (int i = k + 1; i < maxn; i++) {
            for (int j = 0; j < maxn; j++) {
                matrix[i][j] += matrix[k][j];
                matrix[i][j] = (int)matrix[i][j] % 1000;
            }
        }
    }
}

void mpi_GE(int rank, int size) {
    int rows_per_proc = maxn / size;
    int start_row = rank * rows_per_proc;
    int end_row = (rank + 1) * rows_per_proc;
    if (rank == size - 1) {
        end_row = maxn;
    }
    for (int k = 0; k < maxn; k++) {
        if (rank == k / rows_per_proc) {
            for (int j = k + 1; j < maxn; j++) {
                matrix[k][j] /= matrix[k][k];
            }
            matrix[k][k] = 1.0;
        }
        MPI_Bcast(&matrix[k], maxn, MPI_FLOAT, k / rows_per_proc, MPI_COMM_WORLD);

        for (int i = start_row; i < end_row; i++) {
            if (i > k) {
                for (int j = k + 1; j < maxn; j++) {
                    matrix[i][j] -= matrix[i][k] * matrix[k][j];
                }
                matrix[i][k] = 0.0;
            }
        }
    }
}
int main() {
    MPI_Init(NULL,NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        m_set();
    }
    MPI_Bcast(&matrix[0][0], maxn * maxn, MPI_FLOAT, 0, MPI_COMM_WORLD);
    double start, finish;
    start = time();
    mpi_GE(rank, size);
    finish = time();
    if (rank == 0) {
        cout << "parallel: " << (finish - start) / 1000 << "ms\n";
    }
    MPI_Finalize();
    return 0;
}
