#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cuda_runtime.h>

#include <cassert>
#include <exception>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>

#define CE(err)                                                                \
    {                                                                          \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            std::cout << "CUDA error in " << __FUNCTION__ << " (" << __FILE__  \
                      << ":" << __LINE__ << ") - " << cudaGetErrorString(err); \
            std::terminate();                                                  \
        }                                                                      \
    }

constexpr size_t N = 300;

struct payload_t
{
    double a[N];
    double b[N];
    double c[N];
};

template <typename T>
struct fx_k
{
    __host__ __device__ T operator()(T a, T b, T c)
    {
        return a + b + c;
    }
};

void fx()
{
    thrust::host_vector<double> a(N);
    thrust::host_vector<double> b(N);
    thrust::host_vector<double> c(N);

    std::iota(a.begin(), a.end(), 0.0);
    std::iota(b.begin(), b.end(), -static_cast<double>(N));

    // copy to device?
    thrust::device_vector<double> d_a(N);
    thrust::device_vector<double> d_b(N);

    ////////////////////////////////////////////////////////////////////////////
    thrust::device_vector<double> d_c(N);
    thrust::transform(
        d_a.begin(), d_a.end(), d_b.begin(), d_b.begin(), fx_k<double>());
}

void test()
{
    fx();
}

int main()
{
    try
    {
        test();
    }
    catch (std::exception const& ex)
    {
        std::cout << "exception: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
