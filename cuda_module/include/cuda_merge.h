#ifndef CUDA_MERGE_H
#define CUDA_MERGE_H

namespace cuda_merge
{
    template <typename T>
    void mergeCircularBuffer(T *A, int m, T *B, int n, T *C);
}

#endif