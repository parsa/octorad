#include "fx_case.hpp"
#include "fx_compare.hpp"
#include "kernel_cpu.hpp"
#include "kernel_v2.hpp"
#if OCTORAD_HAVE_CUDA
#include "kernel_gpu.hpp"
#endif
#include "util.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

struct scoped_timer
{
    scoped_timer(double& r)
      : start_timepoint(std::chrono::high_resolution_clock::now())
      , value{r}
    {
    }
    ~scoped_timer()
    {
        value = std::chrono::duration<double, std::ratio<1, 1000000000>>(
            std::chrono::high_resolution_clock::now() - start_timepoint)
                    .count();
    }
    double& value;

private:
    std::chrono::high_resolution_clock::time_point start_timepoint;
};

template <typename K>
void run_case_on_kernel(fx_case test_case, K kernel)
{
    fx_args& a = test_case.args;
    kernel(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
        a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c, a.er_i,
        a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz, a.egas, a.tau,
        a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt, a.clightinv);

    bool const success =
        are_ranges_same(test_case.args.egas, test_case.outs.egas, "egas") &&
        are_ranges_same(test_case.args.sx, test_case.outs.sx, "sx") &&
        are_ranges_same(test_case.args.sy, test_case.outs.sy, "sy") &&
        are_ranges_same(test_case.args.sz, test_case.outs.sz, "sz") &&
        are_ranges_same(test_case.args.U, test_case.outs.U, "U");
    if (!success)
    {
        throw formatted_exception("case % code integrity check failed", test_case.index);
    }
}

void run_case(size_t index)
{
    std::printf("***** load case %zd *****\n", index);
    fx_case const test_case = import_case(index);

    std::printf("***** cpu kernel (reference) *****\n");
    double cpu_kernel_duration{};
    {
        scoped_timer{cpu_kernel_duration};
        run_case_on_kernel(test_case, radiation_cpu_kernel);
    }
    std::printf("duration: %g\n", cpu_kernel_duration);

    std::printf("***** cpu kernel (v2) *****\n");
    double v2_kernel_duration{};
    {
        scoped_timer{v2_kernel_duration};
        run_case_on_kernel(test_case, radiation_v2_kernel);
    }
    std::printf("duration: %g\n", v2_kernel_duration);

#if OCTORAD_HAVE_CUDA
    std::printf("***** gpu kernel (ported code) *****\n");
    double gpu_kernel_duration{};
    {
        scoped_timer{gpu_kernel_duration};
        run_case_on_kernel(test_case, radiation_gpu_kernel);
    }
    std::printf("duration: %g\n", gpu_kernel_duration);
#endif
}

int main()
{
    try
    {
        std::size_t const case_id = 78;
        run_case(case_id);
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        //return 1;
    }

    return 0;
}
