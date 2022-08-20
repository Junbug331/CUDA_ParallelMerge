#ifndef CUDA_MERGE_KERNEL_H
#define CUDA_MERGE_KERNEL_H
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h> // __syncthreads()
#include <cuda_runtime.h> // __global__
#include <device_launch_parameters.h> // blockIdx, threadIdx
#include <memory>
#include <stdio.h>

#define ceildiv(a, b) (a+b-1)/b
#define min(a, b) a < b ? a : b
#define max(a, b) a > b ? a : b
#define TILE_SIZE 1024
#define cudaErrChk(ans) { gpuAssert(ans, __FILE__, __LINE__, true); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s file: %s, line: %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


namespace cuda_merge
{
    template <typename T>
    __device__ int coRank(int k, T *A, int m, T *B, int n)
    {
        int i = min(k, m);
        int j = k - i;
        int i_low =  max(0, k-n);
        int j_low = max(0, k-m);
        int delta;
        bool active = true;
        while(active)
        {
            if (i > 0 && j < n && A[i-1] > B[j])
            {
                delta = (i - i_low + 1) >> 1;
                j_low = j;
                i -= delta;
                j += delta;
            }
            else if (j > 0 && i < m && B[j-1] >= A[i])
            {
                delta = (j - j_low + 1) >> 1;
                i_low = i;
                i += delta;
                j -= delta;
            }
            else
                active = false;
        }
        return i;
    }

    template <typename T>
    __device__ int coRankCircular(int k, T *A, int m, T *B, int n, int A_S_start, int B_S_start, int tile_size)
    {
        int i = min(k, m);
        int j = k - i;
        int i_low = max(0, k-n);
        int j_low = max(0, k-m);
        int delta;
        bool active = true;
        while (active)
        {
            if (i > 0 && j < n && A[(A_S_start+i-1)%tile_size] > B[(B_S_start+j)%tile_size])
            {
                delta = ((i - i_low+1) >> 1);
                j_low = j;
                i -= delta;
                j += delta;
            }
            else if (j > 0 && i < m && B[(B_S_start+j-1)%tile_size] > A[(A_S_start+i)%tile_size])
            {
                delta = ((j - j_low+1) >> 1);
                i_low = i;
                j -= delta;
                i += delta;
            }
            else
                active = false;
        }
        return i;

    }

    template <typename T>
    __device__ void mergeSequentialCircular(T *A, int m, T *B, int n, T *C, int A_S_start, int B_S_start, int tile_size)
    {
        /// virtual indexes
        int i = 0;
        int j = 0;
        int k = 0;

        while ((i < m) && (j < n))
        {
            int i_cir = (A_S_start + i) % tile_size;
            int j_cir = (B_S_start + j) % tile_size;
            if (A[i_cir] <= B[j_cir])
            {
                C[k++] = A[i_cir];
                i++;
            }
            else
            {
                C[k++] = B[j_cir];
                j++;
            }
        }
        for (; i<m; i++)
            C[k++] = A[(A_S_start+i)%tile_size];
        for (; j<n; j++)
            C[k++] = B[(B_S_start+j)%tile_size];
    }

    template <typename T>
    __global__ void mergeCircularBufferKernel(T *A, int m, T *B, int n, T *C)
    {
        __shared__ T A_S[TILE_SIZE];
        __shared__ T B_S[TILE_SIZE];

        // Block level C range
        int C_curr = blockIdx.x * ceildiv((m+n), gridDim.x);
        int C_next = min((blockIdx.x + 1) * ceildiv((m+n), gridDim.x), m + n);

        if (threadIdx.x == 0)
        {
            A_S[0] = coRank(C_curr, A, m, B, n);
            A_S[1] = coRank(C_next, A, m, B, n);
        }
        __syncthreads();

        int A_curr = A_S[0];
        int A_next = A_S[1];
        int B_curr = C_curr - A_curr;
        int B_next = C_next - A_next;
        __syncthreads();

        int counter = 0;
        int C_len = C_next - C_curr;
        int A_len = A_next - A_curr;
        int B_len = B_next - B_curr;
        int total_iterations = ceildiv(C_len, TILE_SIZE);
        int C_completed = 0;
        int A_consumed = 0;
        int B_consumed = 0;
        int A_offset = 0;
        int B_offset = 0;
        int A_S_start = 0;
        int B_S_start = 0;
        int A_S_consumed = TILE_SIZE; // set to tile_size for the first iteration
        int B_S_consumed = TILE_SIZE; // same..

        while (counter < total_iterations)
        {
            for (int i=0; i<A_S_consumed; i += blockDim.x)
            {
                if (i + threadIdx.x < A_len - A_consumed && i + threadIdx.x < A_S_consumed)
                {
                    A_S[(A_S_start + (TILE_SIZE-A_S_consumed) + i + threadIdx.x) % TILE_SIZE] =
                            A[A_curr + A_offset + i + threadIdx.x];
                }
            }

            for (int i=0; i<B_S_consumed; i += blockDim.x)
            {
                if (i + threadIdx.x < B_len - B_consumed && i + threadIdx.x < B_S_consumed)
                {
                    B_S[(B_S_start + (TILE_SIZE - B_S_consumed) + i + threadIdx.x) % TILE_SIZE] =
                            B[B_curr + B_offset + i + threadIdx.x];
                }
            }

            A_offset += A_S_consumed;
            B_offset += B_S_consumed;
            __syncthreads();

            /// thread level code
            int c_curr = threadIdx.x * ceildiv(TILE_SIZE, blockDim.x);
            int c_next = (threadIdx.x + 1) * ceildiv(TILE_SIZE, blockDim.x);

            c_curr = (c_curr <= C_len - C_completed) ? c_curr : C_len - C_completed;
            c_next = (c_next <= C_len - C_completed) ? c_next : C_len - C_completed;

            int a_curr = coRankCircular(c_curr,
                                        A_S, min(TILE_SIZE, A_len - A_consumed),
                                        B_S, min(TILE_SIZE, B_len - B_consumed),
                                        A_S_start, B_S_start, TILE_SIZE);
            int b_curr = c_curr - a_curr;
            int a_next = coRankCircular(c_next,
                                        A_S, min(TILE_SIZE, A_len - A_consumed),
                                        B_S, min(TILE_SIZE, B_len - B_consumed),
                                        A_S_start, B_S_start, TILE_SIZE);
            int b_next = c_next - a_next;
            mergeSequentialCircular(A_S, a_next - a_curr, B_S, b_next - b_curr,
                                    C + C_curr + C_completed + c_curr,
                                    A_S_start + a_curr, B_S_start + b_curr, TILE_SIZE);
            counter++;
            A_S_consumed = coRankCircular(min(TILE_SIZE, C_len - C_completed),
                                          A_S, min(TILE_SIZE, A_len - A_consumed),
                                          B_S, min(TILE_SIZE, B_len - B_consumed),
                                          A_S_start, B_S_start, TILE_SIZE);
            B_S_consumed = min(C_len - C_completed, TILE_SIZE) - A_S_consumed;

            C_completed += min(TILE_SIZE, C_len - C_completed);
            A_consumed += A_S_consumed;
            B_consumed = C_completed - A_consumed;

            // update start position
            A_S_start = (A_S_start + A_S_consumed) % TILE_SIZE;
            B_S_start = (B_S_start + B_S_consumed) % TILE_SIZE;
            __syncthreads();
        }
    }

    template <typename T>
    void mergeCircularBufferCUDA(T *A, int m, T *B, int n, T *C)
    {
        auto deletor = [&](T *ptr){ cudaFree(ptr); };
        std::shared_ptr<T> d_A(new T[m], deletor);
        std::shared_ptr<T> d_B(new T[n], deletor);
        std::shared_ptr<T> d_C(new T[m+n], deletor);

        cudaErrChk(cudaMalloc((void **)&d_A, sizeof(T)*m));
        cudaErrChk(cudaMalloc((void **)&d_B, sizeof(T)*n));
        cudaErrChk(cudaMalloc((void **)&d_C, sizeof(T)*(m+n)));

        cudaErrChk(cudaMemcpy(d_A.get(), A, sizeof(T)*m, cudaMemcpyHostToDevice));
        cudaErrChk(cudaMemcpy(d_B.get(), B, sizeof(T)*n, cudaMemcpyHostToDevice));

        int threadPerBlock = 128;
        int blockPerGrid = (m+n + threadPerBlock-1)/threadPerBlock;
        mergeCircularBufferKernel<<<blockPerGrid, threadPerBlock>>>(d_A.get(), m, d_B.get(), n, d_C.get());
        cudaErrChk(cudaMemcpy(C, d_C.get(), sizeof(T)*(m+n), cudaMemcpyDeviceToHost));
    }
}


#endif