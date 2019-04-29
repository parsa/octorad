#include "fx_case.hpp"
#include "kernel_cpu.hpp"
#include "kernel_gpu.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int const N = 1 << 20;

int main()
{
    std::printf("***** load case *****\n");
    fx_case const test_case = import_case(0);

    std::printf("***** run kernel *****\n");

    {
        fx_args a = test_case.args;
        radiation_cpu_kernel(a.er_i, a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx,
            a.sy, a.sz, a.egas, a.tau, a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc,
            a.dt, a.clightinv);
    }
    //int const num_threads = 8;
    //std::thread threads[num_threads];
    //
    //for (auto& t : threads)
    //{
    //    t = std::thread(launch_kernel);
    //}
    //
    //for (auto& t : threads)
    //{
    //    t.join();
    //}

    //cudaDeviceReset();

    return 0;
}
