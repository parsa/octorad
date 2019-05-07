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
bool check_run_result(fx_case test_case, K kernel)
{
    fx_args& a = test_case.args;
    kernel(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
        a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c, a.er_i,
        a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz, a.egas, a.tau,
        a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt, a.clightinv);

    bool result =
        are_ranges_same(test_case.args.egas, test_case.outs.egas, "egas") &&
        are_ranges_same(test_case.args.sx, test_case.outs.sx, "sx") &&
        are_ranges_same(test_case.args.sy, test_case.outs.sy, "sy") &&
        are_ranges_same(test_case.args.sz, test_case.outs.sz, "sz") &&
        are_ranges_same(test_case.args.U, test_case.outs.U, "U");
    return result;
}

bool check_case(size_t index)
{
    std::printf("***** load case %zd *****\n", index);
    fx_case const test_case = import_case(index);

    std::printf("***** cpu kernel (reference) *****\n");

    if (!check_run_result(test_case, radiation_cpu_kernel))
    {
        std::printf("case %zd code integrity check failed.\n", index);
        return false;
    }
    std::printf("case %zd code integrity check passed.\n", index);

    std::printf("***** cpu kernel (v2) *****\n");

    if (!check_run_result(test_case, radiation_v2_kernel))
    {
        std::printf("case %zd code integrity check failed.\n", index);
        return false;
    }
    std::printf("case %zd code integrity check passed.\n", index);

#if OCTORAD_HAVE_CUDA
    std::printf("***** gpu kernel (ported code) *****\n");
    if (!check_run_result(test_case, radiation_gpu_kernel))
    {
        std::printf("case %zd code integrity check failed.\n", index);
        return false;
    }
    std::printf("case %zd code integrity check passed.\n", index);
#endif

    return true;
}

int main()
{
    try
    {
        //check_case(78);
        constexpr std::size_t case_count = 13140;
        for (std::size_t i = 0 ; i < case_count / 100; ++i)
        {
            if (!check_case(i))
            {
                return 1;
            }
        }
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        //return 1;
    }

    return 0;
}
