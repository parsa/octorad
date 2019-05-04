#include "fx_case.hpp"
#include "kernel_cpu.hpp"
#if OCTORAD_HAVE_CUDA
#include "kernel_gpu.hpp"
#endif

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

// NOTE: this comparison is too loose.
// possible reference for improvement: https://stackoverflow.com/q/17333
bool almost_equal(double const a, double const b)
{
    double const diff = std::abs(a - b);
    //double const largest = std::max(std::abs(a), std::abs(b));
    //return diff < largest * std::numeric_limits<double>::epsilon();
    //return diff < 1e-5;
    return diff <= std::numeric_limits<double>::epsilon();
}

bool are_ranges_same(std::vector<double> const& r1,
    std::vector<double> const& r2, std::string const var_name)
{
    //for (size_t i = 0; i < 10; ++i)
    //{
    //    std::printf("**DEBUG** %s[%zd]: %g, %g\n", var_name.c_str(), i, r1[i], r2[i]);
    //}

    if (!std::equal(r1.begin(), r1.end(), r2.begin(), almost_equal))
    {
        auto mism =
            std::mismatch(r1.begin(), r1.end(), r2.begin(), almost_equal);
        auto mism_index = std::distance(r1.begin(), mism.first);

        std::printf("mismatch in %s[%zd]: %g != %g.\n", var_name.c_str(),
            mism_index, *mism.first, *mism.second);

        size_t mism_count = 0;
        for (; mism.first != r1.end(); ++mism_count)
        {
            mism = std::mismatch(
                ++mism.first, r1.end(), ++mism.second, almost_equal);
        }
        std::printf(
            "mismatch count in %s: %zd.\n", var_name.c_str(), mism_count);
        return false;
    }
    //std::printf("identical values of %s.\n", var_name.c_str());
    return true;
}

bool are_ranges_same(std::array<std::vector<double>, NRF> const& r1,
    std::array<std::vector<double>, NRF> const& r2, std::string const var_name)
{
    for (std::size_t i = 0; i < NRF; ++i)
    {
        std::string var_name_i = var_name + "[" + std::to_string(i) + "]";
        if (!are_ranges_same(r1[i], r2[i], var_name_i))
        {
            return false;
        }
    }
    //std::printf("identical values of %s.\n", var_name.c_str());
    return true;
}

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
        check_case(78);
        //constexpr std::size_t case_count = 13140;
        //for (std::size_t i = 0 ; i < case_count; ++i)
        //{
        //    if (!check_case(i))
        //    {
        //        return 1;
        //    }
        //}
    }
    catch (std::exception const& e)
    {
        std::printf("exception thrown: %s\n", e.what());
        //return 1;
    }

    return 0;
}
