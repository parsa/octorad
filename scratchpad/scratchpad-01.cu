#include <vector>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void abort_if_cuda_error(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::printf("cuda error: %s\n", cudaGetErrorString(err));
        std::abort();
    }
}

void abort_if_cuda_error()
{
    cudaError_t err = cudaGetLastError();
    abort_if_cuda_error(err);
}

__global__ void k(double* a, std::size_t a_size)
{
    auto thread_id = threadIdx.x;
    std::printf("value: %g\n", a[thread_id]);
}

int main()
{
    std::vector<double> a(5, 20.0);
    double* d_a{};
    abort_if_cuda_error(cudaMalloc((void**) &d_a, a.size() * sizeof(double)));
    abort_if_cuda_error(cudaMemcpy(d_a, &a[0], a.size() * sizeof(double), cudaMemcpyHostToDevice));

    k<<<1, 5>>>(d_a, a.size());
    abort_if_cuda_error();
    return 0;
}
