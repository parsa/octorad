#pragma once

#include <array>
#include <cstddef>
#include <exception>
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
T* device_alloc(std::size_t const size)
{
    T* device_ptr;
    throw_if_cuda_error(cudaMalloc((void**) &device_ptr, size));
    return device_ptr;
}

template <typename T>
T* device_copy_from_host(T* const device_ptr, std::vector<T> const& v)
{
    throw_if_cuda_error(cudaMemcpy(
        device_ptr, &v[0], v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device_ptr;
}

template <typename T, std::size_t N>
T* device_copy_from_host(
    T* const device_ptr, std::array<std::vector<T>, N> const& a)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        throw_if_cuda_error(cudaMemcpy(device_ptr + i * a[0].size(), &a[i][0],
            a[i].size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return device_ptr;
}

template <typename T>
T* device_alloc_copy_from_host(std::vector<T> const& v)
{
    T* const device_ptr = device_alloc<T>(v.size() * sizeof(T));
    device_copy_from_host<T>(device_ptr, v);
    return device_ptr;
}

template <typename T, std::size_t N>
T* device_alloc_copy_from_host(std::array<std::vector<T>, N> const& a)
{
    T* const device_ptr = device_alloc<T>(N * a[0].size() * sizeof(T));
    device_copy_from_host<T, N>(device_ptr, v);
    return device_ptr;
}

template <typename T>
void device_copy_to_host(T const* const device_ptr, std::vector<T>& v)
{
    throw_if_cuda_error(cudaMemcpy(
        &v[0], device_ptr, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, std::size_t N>
void device_copy_to_host(
    T const* const device_ptr, std::array<std::vector<T>, N>& a)
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
    throw_if_cuda_error(cudaFree((void*) device_ptr));
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
