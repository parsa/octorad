#include <math.h>
#include <stdio.h>
#include "cuda-gpuinfo.h"

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r) * (num) + (c))

/*
 * `step_kernel_mod` is currently a direct copy of the CPU reference solution
 * `step_kernel_ref` below. Accelerate it to run as a CUDA kernel.
 */

__global__ void step_kernel_mod(
    int ni, int nj, float fact, float* temp_in, float* temp_out)
{
    int i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;

    size_t i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y + 1;

    // loop over all points in domain (except boundary)
    if (j < nj - 1 && i < ni - 1)
    {
        // find indices into linear memory
        // for central point and neighbours
        i00 = I2D(ni, i, j);
        im10 = I2D(ni, i - 1, j);
        ip10 = I2D(ni, i + 1, j);
        i0m1 = I2D(ni, i, j - 1);
        i0p1 = I2D(ni, i, j + 1);

        // evaluate derivatives
        d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
        d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];

        // update temperatures
        temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
    }
}

void step_kernel_ref(
    int ni, int nj, float fact, float* temp_in, float* temp_out)
{
    int i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;

    // loop over all points in domain (except boundary)
    for (int j = 1; j < nj - 1; j++)
    {
        for (int i = 1; i < ni - 1; i++)
        {
            // find indices into linear memory
            // for central point and neighbours
            i00 = I2D(ni, i, j);
            im10 = I2D(ni, i - 1, j);
            ip10 = I2D(ni, i + 1, j);
            i0m1 = I2D(ni, i, j - 1);
            i0p1 = I2D(ni, i, j + 1);

            // evaluate derivatives
            d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
            d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];

            // update temperatures
            temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
        }
    }
}

void abortIfError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
        abort();
    }
}

int main()
{
    gpu_info_print();

    int istep;
    int nstep = 200;    // number of time steps

    // Specify our 2D dimensions
    int const ni = 200;
    int const nj = 100;
    float tfac = 8.418e-5;    // thermal diffusivity of silver

    float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp;

    const int size = ni * nj * sizeof(float);

    temp1_ref = (float*) malloc(size);
    temp2_ref = (float*) malloc(size);
    abortIfError(cudaMallocManaged(&temp1, size));
    abortIfError(cudaMallocManaged(&temp2, size));

    // Initialize with random data
    for (int i = 0; i < ni * nj; ++i)
    {
        temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] =
            (float) rand() / (float) (RAND_MAX / 100.0f);
    }

    // Execute the CPU-only reference version
    for (istep = 0; istep < nstep; istep++)
    {
        step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

        // swap the temperature pointers
        temp_tmp = temp1_ref;
        temp1_ref = temp2_ref;
        temp2_ref = temp_tmp;
    }

    dim3 threadsPerBlock(32, 32, 1);
    dim3 blockCount((ni + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (nj + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
    // Execute the modified version using same data
    for (istep = 0; istep < nstep; istep++)
    {
        step_kernel_mod<<<blockCount, threadsPerBlock>>>(
            ni, nj, tfac, temp1, temp2);
        abortIfError(cudaGetLastError());

        abortIfError(cudaDeviceSynchronize());
        // swap the temperature pointers
        temp_tmp = temp1;
        temp1 = temp2;
        temp2 = temp_tmp;
    }

    float maxError = 0;
    // Output should always be stored in the temp1 and temp1_ref at this point
    for (int i = 0; i < ni * nj; ++i)
    {
        if (abs(temp1[i] - temp1_ref[i]) > maxError)
        {
            maxError = abs(temp1[i] - temp1_ref[i]);
        }
    }

    // Check and see if our maxError is greater than an error bound
    if (maxError > 0.0005f)
        printf(
            "Problem! The Max Error of %.5f is NOT within acceptable bounds.\n",
            maxError);
    else
        printf(
            "The Max Error of %.5f is within acceptable bounds.\n", maxError);

    free(temp1_ref);
    free(temp2_ref);
    abortIfError(cudaFree(temp1));
    abortIfError(cudaFree(temp2));

    return 0;
}
