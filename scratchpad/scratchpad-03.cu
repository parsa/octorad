#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <vector>

#include "cuda_runtime.h"

inline void throw_if_cuda_error(cudaError_t err = cudaGetLastError())
{
    if (err != cudaSuccess)
    {
        std::stringstream err_ss;
        err_ss << "cuda error: " << cudaGetErrorString(err);
        throw std::runtime_error(err_ss.str());
    }
}

template <typename T>
T* host_pinned_alloc(std::size_t const size)
{
    // cudaMemAttachGlobal: Memory can be accessed by any stream on any device
    T* device_ptr;
    throw_if_cuda_error(
        cudaMallocHost((void**) &device_ptr, size, cudaMemAttachGlobal));
    return device_ptr;
}

template <typename T>
void device_free(T const* const device_ptr)
{
    throw_if_cuda_error(cudaFreeHost((void*) device_ptr));
}

template <class T>
class cuda_host_allocator
{
public:
    using size_type = std::size_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using const_reference = const T&;

    cuda_host_allocator() {}

    template <class U>
    cuda_host_allocator(cuda_host_allocator<U> const&)
    {
    }

    pointer allocate(size_type num, void const* = 0)
    {
        return host_pinned_alloc<value_type>(num * sizeof(T));
    }

    void deallocate(pointer p, size_type num)
    {
        device_free(p);
    }
};

template <class T1, class T2>
bool operator==(
    cuda_host_allocator<T1> const&, cuda_host_allocator<T2> const&) throw()
{
    return true;
}
template <class T1, class T2>
bool operator!=(
    cuda_host_allocator<T1> const&, cuda_host_allocator<T2> const&) throw()
{
    return false;
}

void test()
{
    std::vector<double, cuda_host_allocator<double>> v(100);
    std::iota(v.begin(), v.end(), 0);
    std::copy(
        v.begin(), v.end(), std::ostream_iterator<double>(std::cout, ", "));
}

int main()
{
    test();

    return 0;
}
