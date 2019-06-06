#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <exception>

#define CE(err)                                                                \
    {                                                                          \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            std::printf("CUDA error in %s (%s:%d) - %s",                       \
                __FUNCTION__,                                                  \
                __FILE__,                                                      \
                __LINE__,                                                      \
                cudaGetErrorString(err));                                      \
            std::terminate();                                                  \
        }                                                                      \
    }

constexpr size_t operator"" KiB(size_t size)
{
    return size * 1024;
}

__global__ void static_reverse(int* const d, int const n)
{
    __shared__ int s[64];
    int const t = threadIdx.x;
    int const tr = n - t - 1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

__global__ void dynamic_reverse(int* d, int n)
{
    extern __shared__ int s[];
    int const t = threadIdx.x;
    int const tr = n - t - 1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

int main()
{
    constexpr int N = 64;

    static_assert(N <= 1024,
        "N can be at most equal to the maximum number of threads in a block");

    constexpr size_t shared_memory_capacity = 48KiB;
    static_assert(N * sizeof(int) <= shared_memory_capacity,
        "Shared memory array size exceeds shared memory capacity");

    int a[N];
    int r[N];
    int d[N];

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        r[i] = N - i - 1;
        d[i] = 0;
    }

    int* d_d;
    CE(cudaMalloc(&d_d, N * sizeof(int)));

    // run version with static shared memory
    CE(cudaMemcpy(d_d, a, N * sizeof(int), cudaMemcpyHostToDevice));
    static_reverse<<<1, N>>>(d_d, N);
    CE(cudaGetLastError());
    CE(cudaMemcpy(d, d_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
    {
        if (d[i] != r[i])
        {
            std::printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
        }
    }

    // run dynamic shared memory version
    CE(cudaMemcpy(d_d, a, N * sizeof(int), cudaMemcpyHostToDevice));
    dynamic_reverse<<<1, N, N * sizeof(int)>>>(d_d, N);
    CE(cudaGetLastError());
    CE(cudaMemcpy(d, d_d, N * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++)
    {
        if (d[i] != r[i])
        {
            std::printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
        }
    }
}
