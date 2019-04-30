#include <stdio.h>
#include <thread>

int const N = 1 << 20;

__global__ void kernel(float* x, int n)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        x[i] = sqrt(pow(3.141592, (int)i));
    }
}

void launch_kernel()
{
    float* data;
    cudaMalloc(&data, N * sizeof(float));

    //int n_ = N;
    //void* args[] = {&data, &n_};
    //cudaLaunchKernel((void const*)&kernel, 1, 64, args);
    kernel<<<1, 64>>>(data, N);

    cudaStreamSynchronize(0);
}

int main()
{
    int const num_threads = 8;
    std::thread threads[num_threads];

    for (auto& t : threads)
    {
        t = std::thread(launch_kernel);
    }

    for (auto& t : threads)
    {
        t.join();
    }

    cudaDeviceReset();

    return 0;
}
