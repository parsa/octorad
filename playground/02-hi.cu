#include <math.h>
#include <stdio.h>

__device__ char const* const str = "HELLO WORLD!";
constexpr size_t str_length = 12;

__global__ void hello()
{
    printf("%c\n", str[threadIdx.x % str_length]);
}

int main()
{
    constexpr size_t thread_count = str_length;
    constexpr size_t block_count = 1;
    hello<<<block_count, thread_count>>>();
    cudaDeviceSynchronize();

    return 0;
}
