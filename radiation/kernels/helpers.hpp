#pragma once

#include <array>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <memory>
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

inline void fail_if_cuda_error(cudaError_t err = cudaGetLastError())
{
    if (err != cudaSuccess)
    {
        std::printf("cuda error: %s", cudaGetErrorString(err));
        std::terminate();
    }
}

template <typename T>
T* device_alloc(std::size_t const count = 1)
{
    T* device_ptr;
    throw_if_cuda_error(cudaMalloc((void**) &device_ptr, count * sizeof(T)));
    return device_ptr;
}

template <typename T>
T* device_copy_from_host_async(T* const device_ptr, T* const payload,
    std::size_t const count, cudaStream_t stream)
{
    throw_if_cuda_error(cudaMemcpyAsync(device_ptr, payload, count * sizeof(T),
        cudaMemcpyHostToDevice, stream));
    return device_ptr;
}

template <typename T>
T* device_copy_from_host(
    T* const device_ptr, T* const payload, std::size_t const count)
{
    throw_if_cuda_error(cudaMemcpy(
        device_ptr, payload, count * sizeof(T), cudaMemcpyHostToDevice));
    return device_ptr;
}

template <typename T, typename Allocator>
T* device_copy_from_host(
    T* const device_ptr, std::vector<T, Allocator> const& v)
{
    throw_if_cuda_error(cudaMemcpy(
        device_ptr, &v[0], v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device_ptr;
}

template <typename T, typename Allocator, std::size_t N>
T* device_copy_from_host(
    T* const device_ptr, std::array<std::vector<T, Allocator>, N> const& a)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        throw_if_cuda_error(cudaMemcpy(device_ptr + i * a[0].size(), &a[i][0],
            a[i].size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return device_ptr;
}

template <typename T, typename Allocator>
T* device_alloc_copy_from_host(std::vector<T, Allocator> const& v)
{
    T* const device_ptr = device_alloc<T>(v.size());
    device_copy_from_host<T>(device_ptr, v);
    return device_ptr;
}

template <typename T, typename Allocator, std::size_t N>
T* device_alloc_copy_from_host(
    std::array<std::vector<T, Allocator>, N> const& a)
{
    T* const device_ptr = device_alloc<T>(N * a[0].size());
    device_copy_from_host<T, N>(device_ptr, a);
    return device_ptr;
}

template <typename T>
void device_copy_to_host_async(T const* const device_ptr, T* const payload,
    std::size_t const count, cudaStream_t stream)
{
    throw_if_cuda_error(cudaMemcpyAsync(payload, device_ptr, count * sizeof(T),
        cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void device_copy_to_host(
    T const* const device_ptr, T* const payload, std::size_t const count)
{
    throw_if_cuda_error(cudaMemcpy(
        payload, device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, typename Allocator>
void device_copy_to_host(
    T const* const device_ptr, std::vector<T, Allocator>& v)
{
    throw_if_cuda_error(cudaMemcpy(
        &v[0], device_ptr, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, typename Allocator, std::size_t N>
void device_copy_to_host(
    T const* const device_ptr, std::array<std::vector<T, Allocator>, N>& a)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        throw_if_cuda_error(cudaMemcpy(&a[i][0], device_ptr + i * a[i].size(),
            a[i].size() * sizeof(T), cudaMemcpyDeviceToHost));
    }
}

template <typename T>
void device_free(T const* const device_ptr)
{
    fail_if_cuda_error(cudaFree((void*) device_ptr));
}

template <typename T>
T* host_pinned_alloc(std::size_t const count = 1)
{
    // cudaMemAttachGlobal: Memory can be accessed by any stream on any device
    T* device_ptr;
    throw_if_cuda_error(cudaMallocHost(
        (void**) &device_ptr, count * sizeof(T), cudaMemAttachGlobal));
    return device_ptr;
}

template <typename T>
void host_pinned_free(T const* const device_ptr)
{
    fail_if_cuda_error(cudaFreeHost((void*) device_ptr));
}

cudaStream_t device_stream_create()
{
    cudaStream_t strm;
    throw_if_cuda_error(cudaStreamCreate(&strm));

    return strm;
}

void device_stream_destroy(cudaStream_t strm)
{
    fail_if_cuda_error(cudaStreamDestroy(strm));
}

template <typename F>
void launch_kernel(F fx, dim3 grid_dim = dim3(1), dim3 block_dim = dim3(1),
    cudaStream_t default_stream = 0)
{
    throw_if_cuda_error(cudaLaunchKernel(
        (void*) fx, grid_dim, block_dim, nullptr, 0, default_stream));
}

template <typename F, typename... Ts>
void launch_kernel(F fx, dim3 grid_dim = dim3(1), dim3 block_dim = dim3(1),
    cudaStream_t default_stream = 0, Ts... args)
{
    void* kernel_args[sizeof...(args)] = {&args...};
    throw_if_cuda_error(cudaLaunchKernel(
        (void*) fx, grid_dim, block_dim, kernel_args, 0, default_stream));
}

template <class T>
class cuda_host_allocator
{
public:
    using size_type = std::size_t;
    using value_type = T;
    using pointer = T*;
    using reference = T&;
    using const_reference = T const&;

    cuda_host_allocator() {}

    template <class U>
    cuda_host_allocator(cuda_host_allocator<U> const&)
    {
    }

    pointer allocate(size_type num, void const* = 0)
    {
        return host_pinned_alloc<value_type>(num * sizeof(value_type));
    }

    void deallocate(pointer p, size_type num)
    {
        host_pinned_free(p);
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
