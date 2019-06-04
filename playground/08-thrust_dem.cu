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
#include <random>
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

constexpr size_t N = 3'000'000;

void case1()
{
    thrust::host_vector<int> h_vec(N);

    // set values
    std::mt19937 m(0);
    std::uniform_int_distribution<int> d(0, 20);
    thrust::generate(h_vec.begin(), h_vec.end(), [&]() { return d(m); });

    // copy to device?
    thrust::device_vector<int> d_vec;
    d_vec = h_vec;

    ////////////////////////////////////////////////////////////////////////////
    // actual kernel
    thrust::sort(thrust::device, d_vec.begin(), d_vec.end());

    // copy back
    h_vec = d_vec;
}

void case2()
{
    thrust::host_vector<int> h_vec(N);

    // set values
    std::mt19937 m(0);
    std::uniform_int_distribution<int> d(0, 20);
    thrust::generate(h_vec.begin(), h_vec.end(), [&]() { return d(m); });

    thrust::device_vector<int> d_vec;
    d_vec = h_vec;

    ////////////////////////////////////////////////////////////////////////////
    // reduce
    int sum = thrust::reduce(d_vec.begin(), d_vec.end());
}

template <typename T>
struct k1
{
    __host__ __device__ T operator()(T a, T b)
    {
        return a + b;
    }
};

void case3()
{
    thrust::host_vector<int> hv1(N);
    thrust::host_vector<int> hv2(N);

    // set values
    std::mt19937 m(0);
    std::uniform_int_distribution<int> d(0, 20);
    thrust::generate(hv1.begin(), hv1.end(), [&]() { return d(m); });
    thrust::generate(hv2.begin(), hv2.end(), [&]() { return d(m); });

    // copy to device?
    thrust::device_vector<int> dv1 = hv1;
    thrust::device_vector<int> dv2 = hv2;

    ////////////////////////////////////////////////////////////////////////////
    thrust::device_vector<int> dvr(N);
    thrust::transform(
        dv1.begin(), dv1.end(), dv2.begin(), dvr.begin(), k1<int>());
}

void test()
{
    case1();
    case2();
    case3();
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
