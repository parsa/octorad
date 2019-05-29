#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>

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

namespace pcw {
    template <typename T>
    T* alloc(size_t count = 1)
    {
        T* device_ptr;
        CE(cudaMalloc(&device_ptr, sizeof(T) * count));
        return device_ptr;
    }
    template <typename T>
    void copy_to(T* const dest, T const* const src, size_t count = 1)
    {
        CE(cudaMemcpy(dest, src, sizeof(T) * count, cudaMemcpyHostToDevice));
    }
    template <typename T>
    void copy_from(T* const dest, T const* const src, size_t count = 1)
    {
        CE(cudaMemcpy(dest, src, sizeof(T) * count, cudaMemcpyDeviceToHost));
    }
    template <typename T>
    void free(T* const device_ptr)
    {
        CE(cudaFree(device_ptr));
    }
}

constexpr size_t BLOCK_SIZE = 16;

template <typename T>
size_t determine_width(std::initializer_list<std::initializer_list<T>> lst)
{
    size_t wd = 0;
    for (auto const& rw : lst)
    {
        wd = std::max(wd, rw.size());
    }
    return wd;
}

struct matrix_t
{
    size_t width = 0;
    size_t height = 0;
    float* elements = nullptr;

    void operator=(std::initializer_list<std::initializer_list<float>> vals)
    {
        assert(width == determine_width(vals));
        assert(height == vals.size());

        size_t i = 0;
        for (auto const& rw : vals)
        {
            std::fill(std::copy(rw.begin(), rw.end(), elements + width * i),
                elements + width * (i + 1),
                0.0f);
            ++i;
        }
    }

    size_t element_count() const
    {
        return width * height;
    }

    __host__ __device__ float operator()(size_t wd, size_t ht) const
    {
        return elements[wd * width + ht];
    }

    __host__ __device__ float& operator()(size_t wd, size_t ht)
    {
        return elements[wd * width + ht];
    }
};

std::ostream& operator<<(std::ostream& os, matrix_t const& m)
{
    os << "{\n";
    for (size_t i = 0; i < m.height; ++i)
    {
        os << "    {";
        for (size_t j = 0; j < m.width; ++j)
        {
            os << m(i, j) << ", ";
        }
        os << "}\n";
    }
    os << "}\n";

    return os;
}

__global__ void multiply_matrices_kernel(matrix_t const,
    matrix_t const,
    matrix_t);

void multiply_matrices(matrix_t const a, matrix_t const b, matrix_t c)
{
    matrix_t d_a;
    {
        d_a.width = a.width;
        d_a.height = a.height;
        d_a.elements = pcw::alloc<float>(d_a.element_count());
        pcw::copy_to(d_a.elements, a.elements, a.element_count());
    }

    matrix_t d_b;
    {
        d_b.width = b.width;
        d_b.height = b.height;
        d_b.elements = pcw::alloc<float>(d_b.element_count());
        pcw::copy_to(d_b.elements, b.elements, b.element_count());
    }

    matrix_t d_c;
    {
        d_c.width = c.width;
        d_c.height = c.height;
        d_c.elements = pcw::alloc<float>(d_c.element_count());
    }

    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim((unsigned int) (b.width + block_dim.x - 1) / block_dim.x,
        (unsigned int) (a.height + block_dim.y - 1) / block_dim.y);
    multiply_matrices_kernel<<<grid_dim, block_dim>>>(d_a, d_b, d_c);

    {
        pcw::copy_from(c.elements, d_c.elements, c.element_count());
    }

    pcw::free(d_a.elements);
    pcw::free(d_b.elements);
    pcw::free(d_c.elements);
}

__global__ void multiply_matrices_kernel(matrix_t const a,
    matrix_t const b,
    matrix_t c)
{
    size_t const row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t const col = blockIdx.x * blockDim.x + threadIdx.x;

    float c_value = 0.0f;

    for (size_t e = 0; e < a.width; ++e)
    {
        c_value += a(row, e) * b(e, col);
        //std::printf("c_value = %g\n", c_value);
        c(row, col) = c_value;
    }
}

int main()
{
    try
    {
        matrix_t a;
        a.height = 2;
        a.width = 3;
        a.elements = new float[a.element_count()];
        a = {{2, 1, 4}, {0, 1, 1}};
        std::cout << a;

        matrix_t b;
        b.height = 3;
        b.width = 4;
        b.elements = new float[b.element_count()];
        b = {{6, 3, -1, 0}, {1, 1, 0, 4}, {-2, 5, 0, 2}};
        std::cout << b;

        matrix_t c;
        c.height = 2;
        c.width = 4;
        c.elements = new float[c.element_count()];
        std::fill_n(c.elements, c.element_count(), 0.0f);

        multiply_matrices(a, b, c);
        std::cout << c;

        //assert(5 == c.elements[0]);
        //assert(27 == c.elements[1]);
        //assert(-2 == c.elements[2]);
        //assert(12 == c.elements[3]);
        //assert(-1 == c.elements[4]);
        //assert(6 == c.elements[5]);
        //assert(0 == c.elements[6]);
        //assert(6 == c.elements[7]);

        delete[] a.elements;
        delete[] b.elements;
        delete[] c.elements;
    }
    catch (std::exception const& ex)
    {
        std::cout << "exception: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
