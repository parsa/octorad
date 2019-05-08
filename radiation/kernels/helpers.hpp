#pragma once

#include <array>
#include <cstddef>
#include <exception>
#include <sstream>
#include <vector>

#include "cuda_runtime.h"

inline void throw_if_cuda_error(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::stringstream err_ss;
        err_ss << "cuda error: " << cudaGetErrorString(err);
        throw std::runtime_error(err_ss.str());
    }
}

inline void throw_if_cuda_error()
{
    cudaError_t err = cudaGetLastError();
    throw_if_cuda_error(err);
}

template <typename T>
T* alloc_device(std::size_t const size)
{
    T* device_ptr;
    throw_if_cuda_error(cudaMalloc((void**) &device_ptr, size));
    return device_ptr;
}

template <typename T>
T* alloc_copy_device(std::vector<T> const& v)
{
    T* const device_ptr = alloc_device<T>(v.size() * sizeof(T));
    throw_if_cuda_error(cudaMemcpy(
        device_ptr, &v[0], v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return device_ptr;
}

template <typename T, std::size_t N>
T* alloc_copy_device(std::array<std::vector<T>, N> const& a)
{
    T* const device_ptr = alloc_device<T>(N * a[0].size() * sizeof(T));
    for (std::size_t i = 0; i < N; ++i)
    {
        throw_if_cuda_error(cudaMemcpy(device_ptr + i * a[0].size(), &a[i][0],
            a[i].size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    return device_ptr;
}

template <typename T>
void copy_from_device(T const* const device_ptr, std::vector<T>& v)
{
    throw_if_cuda_error(cudaMemcpy(
        &v[0], device_ptr, v.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T, std::size_t N>
void copy_from_device(
    T const* const device_ptr, std::array<std::vector<T>, N>& a)
{
    for (std::size_t i = 0; i < N; ++i)
    {
        throw_if_cuda_error(cudaMemcpy(&a[i][0], device_ptr + i * a[i].size(),
            a[i].size() * sizeof(T), cudaMemcpyDeviceToHost));
    }
}

template <typename T>
void free_device(T const* const device_ptr)
{
    throw_if_cuda_error(cudaFree((void*) device_ptr));
}
