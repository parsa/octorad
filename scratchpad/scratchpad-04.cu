#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>

constexpr size_t BLOCKSIZE_x = 16;
constexpr size_t BLOCKSIZE_y = 16;

constexpr size_t N = 32;
constexpr size_t M = 16;
constexpr size_t W = 4;

//////////////////
// cuda_err_chk //
//////////////////

#define cuda_err_chk(ans)                                                      \
    {                                                                          \
        gpu_assert((ans), __FILE__, __LINE__);                                 \
    }

inline void gpu_assert(cudaError_t code, char const* file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "assert failed: %s %s %dn", cudaGetErrorString(code),
            file, line);

        abort();
    }
}

//////////////
// i_div_up //
//////////////

int i_div_up(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

////////////////////
// test_kernel_3d //
////////////////////

__global__ void test_kernel_3d(cudaPitchedPtr dev_pitched_ptr)
{
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    char* d_ptr = (char*) dev_pitched_ptr.ptr;
    size_t pitch = dev_pitched_ptr.pitch;
    size_t slice_pitch = pitch * N;

    for (int w = 0; w < W; ++w)
    {
        char* slice = d_ptr + w * slice_pitch;
        double* row = (double*) (slice + tid_y * pitch);
        row[tid_x] = row[tid_x] * row[tid_x];
    }
}

//////////
// main //
//////////

int main()
{
    double a[N][M][W];

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            for (int w = 0; w < W; ++w)
            {
                a[i][j][w] = 3.f;
                //printf("row %i column %i depth %i value %f n", i, j, w, a[i][j][w]);
            }
        }
    }

    cudaExtent extent = make_cudaExtent(M * sizeof(double), N, W);

    cudaPitchedPtr d_pitched_ptr;

    cuda_err_chk(cudaMalloc3D(&d_pitched_ptr, extent));

    cudaMemcpy3DParms p = {0};

    p.srcPtr.ptr = a;
    p.srcPtr.pitch = M * sizeof(double);
    p.srcPtr.xsize = M;
    p.srcPtr.ysize = N;
    p.dstPtr.ptr = d_pitched_ptr.ptr;
    p.dstPtr.pitch = d_pitched_ptr.pitch;
    p.dstPtr.xsize = M;
    p.dstPtr.ysize = N;
    p.extent.width = M * sizeof(double);
    p.extent.height = N;
    p.extent.depth = W;
    p.kind = cudaMemcpyHostToDevice;

    cuda_err_chk(cudaMemcpy3D(&p));

    dim3 grid_size(i_div_up(M, BLOCKSIZE_x), i_div_up(N, BLOCKSIZE_y));

    dim3 block_size(BLOCKSIZE_y, BLOCKSIZE_x);

    test_kernel_3d<<<grid_size, block_size>>>(d_pitched_ptr);

    cuda_err_chk(cudaPeekAtLastError());

    cuda_err_chk(cudaDeviceSynchronize());

    p.srcPtr.ptr = d_pitched_ptr.ptr;
    p.srcPtr.pitch = d_pitched_ptr.pitch;
    p.dstPtr.ptr = a;
    p.dstPtr.pitch = M * sizeof(double);
    p.kind = cudaMemcpyDeviceToHost;

    cuda_err_chk(cudaMemcpy3D(&p));

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < M; ++j)
        {
            for (int w = 0; w < W; ++w)
            {
                printf("row %3i column %3i depth %3i value %f\n", i, j, w,
                    a[i][j][w]);
            }
        }
    }

    return 0;
}
