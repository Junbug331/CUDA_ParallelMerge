#include <iostream>
#include <stdlib.h>
#include <memory>
#include <algorithm>
#include <random>
#include <string>
#include <omp.h>
#include "cuda_merge.h"
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

void mergesort(int a[], int i, int j);
void merge(int a[], int i1, int j1, int i2, int j2);

template<typename T>
void mergeCPU(T *A, int m, T *B, int n, T *C)
{
    int i, j, k;
    i = j = k = 0;

    while(i<m && j<n)
    {
        if (A[i] < B[j])
            C[k++] = A[i++];
        else
            C[k++] = B[j++];
    }

    while(i<m) C[k++]=A[i++];
    while(j<n) C[k++]=B[j++];
}
template <typename T>
bool validate(T *C, T *ans, size_t N)
{
    for (size_t i=0; i<N; i++)
    {
        if (C[i] != ans[i])
        {
            return false;
        }
    }
    return true;
}


int main(int argc, char** argv) 
{
    std::random_device dev;
    std::default_random_engine eng(dev());
    std::uniform_int_distribution<int> dist(1, 100);
    int m = 68123;
    int n = 80129;
    std::unique_ptr<int[]> A(new int[m]);
    std::unique_ptr<int[]> B(new int[n]);
    std::unique_ptr<int[]> C(new int[m+n]{0, });
    std::unique_ptr<int[]> ans(new int[m+n]{0, });


    for (int i=0; i<m; i++)
        A[i] = dist(eng);
    for (int i=0; i<n; i++)
        B[i] = dist(eng);

    std::sort(A.get(), A.get()+m);
    std::sort(B.get(), B.get()+n);
    spdlog::stopwatch sw;
    mergeCPU(A.get(), m, B.get(), n, ans.get());
    spdlog::info("CPU merge took {} sec", sw);
    sw.reset();
    cuda_merge::mergeCircularBuffer(A.get(), m, B.get(), n, C.get());
    spdlog::info("GPU merge took {} sec", sw);
    if (validate(C.get(), ans.get(), m+n)) std::cout << "Success\n";
    else std::cout << "Fail\n";

//    for (int i=0; i<m; i++)
//        std::cout << A[i] << ' ';
//    std::cout << '\n';
//
//    for (int i=0; i<n; i++)
//        std::cout << B[i] << ' ';
//    std::cout << '\n';
//
//    std::cout << "cuda: \n";
//    for (int i=0; i<(n+m); i++)
//        std::cout << C[i] << ' ';
//    std::cout << '\n';
//
//    std::cout << "CPU: \n";
//    for (int i=0; i<(n+m); i++)
//        std::cout << ans[i] << ' ';
//    std::cout << '\n';
}


void mergesort(int a[], int i, int j)
{
    int mid;
    if (i < j)
    {
        mid = (i+j)/2;
    }
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            mergesort(a, i, mid);
        }
        #pragma omp section
        {
            mergesort(a, mid+1, j);
        }
    }
    merge(a, i, mid, mid+1, j);
}

void merge(int a[], int i1, int j1, int i2, int j2)
{
    int temp[500];
    int i, j, k;
    i = i1;
    j = i2;
    k = 0;
    while (i <= j1 && j <= j2)
    {
        if (a[i] <= a[j])
            temp[k++] = a[i++];
        else
            temp[k++] = a[j++];
    }
    while(i <= j1) temp[k++] = a[i++];
    while(j <= j2) temp[k++] = a[j++];
    k--;
    std::copy(temp, temp+k, a+i1);
}