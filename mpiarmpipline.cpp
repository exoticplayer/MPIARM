#include<stdio.h>
#include <iostream>
#include <mpi.h>
#include <sys/time.h>
#include <stdlib.h>
#include <pmmintrin.h>
#include <omp.h>
#include<arm_neon.h>
using namespace std;
float** A;
float** B;
float** C;
float** D;
float** E;
float* recv_arr;
int n = 1000;
const int N = 1000;
const int thread_count = 4;
const int task = 1;
float mat[N][N];
void init_mat(float test[][N])
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            test[i][j] = rand() / 100;
}
void print_mat(float mat[][N])
{
    if (N > 16)
        return;
    cout << endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            cout << mat[i][j] << " ";
        cout << endl;
    }
    cout << endl;
}
void init(int n)
{
    A = new float* [n];
    recv_arr = new float[n];
    for (int i = 0; i < n; i++)
        A[i] = new float[n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < i; j++)
            A[i][j] = 0.0;
        A[i][i] = 1.0;
        for (int j = i + 1; j < n; j++)
            A[i][j] = rand();
    }
    for (int i = 1; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            A[i][j] += A[i - 1][j];
    }


}
void deepcopy()
{
    B = new float* [n];
    for (int i = 0; i < n; i++)
        B[i] = new float[n];
    C = new float* [n];
    for (int i = 0; i < n; i++)
        C[i] = new float[n];
    D = new float* [n];
    for (int i = 0; i < n; i++)
        D[i] = new float[n];
    E = new float* [n];
    for (int i = 0; i < n; i++)
        E[i] = new float[n];
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            B[i][j] = A[i][j];
            C[i][j] = A[i][j];
            D[i][j] = A[i][j];
            E[i][j] = A[i][j];
        }
    }
}
void eliminate(float mat[][N], int rank, int num_proc)
{
    int seg = task * num_proc;
    //    计算当前进程的前一进程及下一进程
    int pre_proc = (rank + (num_proc - 1)) % num_proc;
    int next_proc = (rank + 1) % num_proc;
    for (int k = 0; k < N; k++)
    {
        //        判断当前行是否是自己的任务
        if (int((k % seg) / task) == rank)
        {
            for (int j = k + 1; j < N; j++)
                mat[k][j] = mat[k][j] / mat[k][k];
            mat[k][k] = 1.0;
            //            处理完自己的任务后向下一进程发送消息
            MPI_Send(&mat[k], N, MPI_FLOAT, next_proc, 2, MPI_COMM_WORLD);
        }
        else
        {
            //            如果当前行不是当前进程的任务，则接收前一进程的消息
            MPI_Recv(&mat[k], N, MPI_FLOAT, pre_proc, 2,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //            如果当前行不是下一进程的任务，需将消息进行传递
            if (int((k % seg) / task) != next_proc)
                MPI_Send(&mat[k], N, MPI_FLOAT, next_proc, 2, MPI_COMM_WORLD);
        }
        for (int i = k + 1; i < N; i++)
        {
            if (int((i % seg) / task) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    mat[i][j] = mat[i][j] - mat[i][k] * mat[k][j];
                mat[i][k] = 0.0;
            }
        }
    }
}
void reset_mat(float** mat, float** test)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            mat[i][j] = test[i][j];
}
int main(int argc, char* argv[])
{
    init(n);
    deepcopy();
    int rank, num_proc;
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Get_processor_name(processor_name, &namelen);
    timeval t_start;
    timeval t_end;
    int seg = task * num_proc;
    if (rank == 0)
    {
        init_mat(mat);
        gettimeofday(&t_start, NULL);
        //        在0号进程进行任务划分
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Send(&mat[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        eliminate(mat, rank, num_proc);
        //        处理完0号进程自己的任务后需接收其他进程处理之后的结果
        for (int i = 0; i < N; i++)
        {
            int flag = (i % seg) / task;
            if (flag == rank)
                continue;
            else
                MPI_Recv(&mat[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        gettimeofday(&t_end, NULL);
        cout << "Block MPI LU with SSE and OpenMP time cost: "
            << 1000 * (t_end.tv_sec - t_start.tv_sec) +
            0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
        //print_mat(mat);
    }
    else
    {
        //        非0号进程先接收任务
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Recv(&mat[i + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        eliminate(mat, rank, num_proc);
        //        处理完后向零号进程返回结果
        for (int i = task * rank; i < N; i += seg)
        {
            for (int j = 0; j < task && i + j < N; j++)
                MPI_Send(&mat[i + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
    return 0;


}