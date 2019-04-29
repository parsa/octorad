#include "kernel_gpu.hpp"

__global__ void kernel(float* x, int n)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        x[i] = sqrt(pow(3.141592, (int) i));
    }
}

void launch_kernel_gpu()
{
    //float* data;
    //cudaMalloc(&data, N * sizeof(float));

    //int n_ = N;
    //void* args[] = {&data, &n_};
    //cudaLaunchKernel((void const*)&kernel, 1, 64, args);
    //kernel<<<1, 64>>>(data, N);

    //cudaStreamSynchronize(0);
}
