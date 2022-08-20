#include "cuda_merge.h"
#include "cuda_merge_kernel.h"

namespace cuda_merge
{
    template <typename T>
    void mergeCircularBuffer(T* A, int m, T *B, int n, T *C)
    {
        mergeCircularBufferCUDA(A, m, B, n, C);
    }
}
template void cuda_merge::mergeCircularBuffer<int>(int *, int, int *, int, int *);
template void cuda_merge::mergeCircularBuffer<float>(float *, int, float *, int, float *);
template void cuda_merge::mergeCircularBuffer<double>(double *, int, double *, int, double *);
template void cuda_merge::mergeCircularBuffer<short>(short*, int ,short*, int, short*);
template void cuda_merge::mergeCircularBuffer<unsigned short>(unsigned short*, int ,unsigned short*, int, unsigned short*);
template void cuda_merge::mergeCircularBuffer<unsigned int>(unsigned int*, int ,unsigned int*, int, unsigned int*);
template void cuda_merge::mergeCircularBuffer<long>(long*, int ,long*, int, long*);
template void cuda_merge::mergeCircularBuffer<unsigned long>(unsigned long*, int ,unsigned long*, int, unsigned long*);
template void cuda_merge::mergeCircularBuffer<long long>(long long*, int ,long long*, int, long long*);
template void cuda_merge::mergeCircularBuffer<unsigned long long>(unsigned long long*, int ,unsigned long long*, int, unsigned long long*);