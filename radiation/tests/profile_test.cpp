#include "kernels/kernel_cpu.hpp"
#include "kernels/kernel_v2.hpp"
#if OCTORAD_HAVE_CUDA
#include "kernels/kernel_gpu.hpp"
#endif
#include "utils/fx_case.hpp"
#include "utils/fx_compare.hpp"
#include "utils/scoped_timer.hpp"
#include "utils/util.hpp"

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

template <typename K>
void run_case_on_kernel(octotiger::fx_case test_case, K kernel)
{
    octotiger::fx_args& a = test_case.args;
    kernel(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
        a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c, a.er_i,
        a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz, a.egas, a.tau,
        a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt, a.clightinv);

    bool const success =
        octotiger::are_ranges_same(test_case.args.egas, test_case.outs.egas, "egas") &&
        octotiger::are_ranges_same(test_case.args.sx, test_case.outs.sx, "sx") &&
        octotiger::are_ranges_same(test_case.args.sy, test_case.outs.sy, "sy") &&
        octotiger::are_ranges_same(test_case.args.sz, test_case.outs.sz, "sz") &&
        octotiger::are_ranges_same(test_case.args.U, test_case.outs.U, "U");
    if (!success)
    {
        throw octotiger::formatted_exception(
            "case % code integrity check failed", test_case.index);
    }
}

void run_case(std::size_t index)
{
    std::printf("***** load case %zd *****\n", index);
    octotiger::fx_case const test_case = octotiger::import_case(index);

    std::printf("***** cpu kernel (reference) *****\n");
    double cpu_kernel_duration{};
    {
        octotiger::radiation_cpu_kernel krnl(test_case.data_size);
        scoped_timer<double>{cpu_kernel_duration};
        run_case_on_kernel(test_case, std::move(krnl));
    }
    std::printf("duration: %g\n", cpu_kernel_duration);

    std::printf("***** cpu kernel (v2) *****\n");
    double v2_kernel_duration{};
    {
        octotiger::radiation_v2_kernel krnl(test_case.data_size);
        scoped_timer<double>{v2_kernel_duration};
        run_case_on_kernel(test_case, std::move(krnl));
    }
    std::printf("duration: %g\n", v2_kernel_duration);

#if OCTORAD_HAVE_CUDA
    std::printf("***** gpu kernel (ported code) *****\n");
    double gpu_kernel_duration{};
    {
        octotiger::radiation_gpu_kernel krnl(test_case.data_size);
        scoped_timer<double>{gpu_kernel_duration};
        run_case_on_kernel(test_case, std::move(krnl));
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
        return 1;
    }

    return 0;
}
