#pragma once

#include <cstddef>
#include <cstdio>
#include <exception>
#include <memory>
#include <sstream>

#include "cuda_runtime.h"

namespace copy_policy {
    struct synchronous_policy
    {
    };
    struct asynchronous_policy
    {
    };
    constexpr synchronous_policy sync;
    constexpr asynchronous_policy async;
}

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

template <typename T, typename Policy>
T* device_copy_from_host(Policy, T* const device_ptr, T* const payload,
    cudaStream_t stream, std::size_t const count)
{
    static_assert(sizeof(Policy) == 0, "Invalid execution policy specified.");
}

template <typename T>
T* device_copy_from_host(copy_policy::asynchronous_policy, T* const device_ptr,
    T* const payload, cudaStream_t stream, std::size_t const count = 1)
{
    throw_if_cuda_error(cudaMemcpyAsync(device_ptr, payload, count * sizeof(T),
        cudaMemcpyHostToDevice, stream));
    return device_ptr;
}

template <typename T>
T* device_copy_from_host(copy_policy::synchronous_policy,

    T* const device_ptr, T* const payload, std::size_t const count)
{
    throw_if_cuda_error(cudaMemcpy(
        device_ptr, payload, count * sizeof(T), cudaMemcpyHostToDevice));
    return device_ptr;
}

template <typename T, typename It>
T* device_copy_from_host(copy_policy::synchronous_policy, T* const device_ptr,
    It const& begin, std::size_t size)
{
    throw_if_cuda_error(cudaMemcpy(
        device_ptr, begin, size * sizeof(T), cudaMemcpyHostToDevice));
    return device_ptr;
}
template <typename T, typename Policy>
void device_copy_to_host(Policy, T const* const device_ptr, T* const payload,
    cudaStream_t stream, std::size_t const count = 1)
{
    static_assert(sizeof(Policy) == 0, "Invalid execution policy specified.");
}
template <typename T>
void device_copy_to_host(copy_policy::asynchronous_policy,
    T const* const device_ptr, T* const payload, cudaStream_t stream,
    std::size_t const count = 1)
{
    throw_if_cuda_error(cudaMemcpyAsync(payload, device_ptr, count * sizeof(T),
        cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void device_copy_to_host(copy_policy::synchronous_policy,
    T const* const device_ptr, T* const payload, std::size_t const count)
{
    throw_if_cuda_error(cudaMemcpy(
        payload, device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, typename It>
void device_copy_to_host(copy_policy::synchronous_policy,
    T const* const device_ptr, It begin, std::size_t size)
{
    throw_if_cuda_error(cudaMemcpy(
        begin, device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
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

void device_stream_sync(cudaStream_t strm)
{
    fail_if_cuda_error(cudaStreamSynchronize(strm));
}

template <typename Fx>
void launch_kernel(Fx fx, dim3 grid_dim = dim3(1), dim3 block_dim = dim3(1),
    cudaStream_t stream = 0)
{
    // dynamic shared memory available to each thread block
    constexpr std::size_t shared_memory_size = 0;
    throw_if_cuda_error(cudaLaunchKernel(
        (void*) fx, grid_dim, block_dim, nullptr, shared_memory_size, stream));
}

template <typename Fx, typename... Ts>
void launch_kernel(Fx fx, dim3 grid_dim = dim3(1), dim3 block_dim = dim3(1),
    cudaStream_t stream = 0, Ts... args)
{
    void* kernel_args[sizeof...(args)] = {&args...};
    // dynamic shared memory available to each thread block
    constexpr std::size_t shared_memory_size = 0;
    throw_if_cuda_error(cudaLaunchKernel((void*) fx, grid_dim, block_dim,
        kernel_args, shared_memory_size, stream));
}

template <typename Fx>
void enqueue_host_fx(Fx fx, cudaStream_t stream = 0, void* user_data = nullptr)
{
    throw_if_cuda_error(cudaLaunchHostFunc(stream, fx, user_data));
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
