#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

using duration_t = unsigned long long;

constexpr std::size_t SHARED_MEM_CAPACITY = 49152;
constexpr std::size_t ITERATIONS = 10;

#define CE(err)                                                                \
    {                                                                          \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            std::stringstream err_ss;                                          \
            err_ss << "CUDA error in " << __FUNCTION__ << " (" << __FILE__     \
                   << ":" << __LINE__ << ") - " << cudaGetErrorString(err);    \
            throw std::runtime_error(err_ss.str());                            \
        }                                                                      \
    }

template <std::size_t Count, std::size_t Padding>
struct foo
{
    float d[Count];
    float padding[Padding];
};

template <std::size_t Count>
struct foo<Count, 0>
{
    float d[Count];
};

__global__ void k1(duration_t* const duration)
{
    using subject_t = foo<4, 3>;
    constexpr std::size_t SHARED_MEM_SUBJECT_CAPACITY =
        SHARED_MEM_CAPACITY / sizeof(subject_t);

    __shared__ subject_t arr[SHARED_MEM_SUBJECT_CAPACITY];
    duration_t start_time = clock();

    ++arr[threadIdx.x].d[0];

    duration_t end_time = clock();
    *duration = end_time - start_time;
}

__global__ void k2(duration_t* const duration, std::size_t factor)
{
    using subject_t = double;
    constexpr std::size_t SHARED_MEM_SUBJECT_CAPACITY =
        SHARED_MEM_CAPACITY / sizeof(subject_t);

    __shared__ subject_t arr[SHARED_MEM_SUBJECT_CAPACITY];
    duration_t start_time = clock();

    ++arr[threadIdx.x * factor];

    duration_t end_time = clock();
    *duration = end_time - start_time;
}

void run_k1(duration_t* const d_duration)
{
    std::cout << "===== k1 =====\n";
    duration_t duration = 0;
    for (std::size_t i = 0; i < ITERATIONS; ++i)
    {
        k1<<<1, 256>>>(d_duration);
        CE(cudaMemcpy(
            &duration, d_duration, sizeof(duration_t), cudaMemcpyDeviceToHost));

        std::cout << "Duration: " << (duration / 100.) << '\n';
    }
}

void run_k2(duration_t* const d_duration, std::size_t factor)
{
    std::cout << "===== k2: " << factor << " =====\n";
    duration_t duration = 0;
    for (std::size_t i = 0; i < ITERATIONS; ++i)
    {
        k2<<<1, 256>>>(d_duration, factor);
        CE(cudaMemcpy(
            &duration, d_duration, sizeof(duration_t), cudaMemcpyDeviceToHost));

        std::cout << "Duration: " << (duration / 100.) << '\n';
    }
}

void run_kn()
{
    duration_t* d_duration;
    CE(cudaMalloc(&d_duration, sizeof(duration_t)));

    run_k1(d_duration);
    for (std::size_t factor = 0; factor < 20; ++factor)
    {
        run_k2(d_duration, factor);
    }

    CE(cudaFree(d_duration));

    CE(cudaDeviceReset());
}

int main()
{
    try
    {
        run_kn();
    }
    catch (std::exception const& ex)
    {
        std::cout << "exception: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
