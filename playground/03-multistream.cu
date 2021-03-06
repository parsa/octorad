int const N = 1 << 20;

__global__ void kernel(float* x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
    {
        x[i] = sqrt(pow(3.141592, i));
    }
}

int main()
{
    int const num_streams = 8;
    cudaStream_t streams[num_streams];

    float* data[num_streams];

    for (int i = 0; i < num_streams; i++)
    {
        cudaStreamCreate(&streams[i]);

        cudaMalloc(&data[i], N * sizeof(float));

        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    return 0;
}
