#include "fx_case.hpp"
#include "kernel_cpu.hpp"
#include "kernel_gpu.hpp"

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

int const N = 1 << 20;

bool are_ranges_same(std::vector<double> const& r1,
    std::vector<double> const& r2, std::string const var_name)
{
    for (size_t i = 0; i < 10; ++i)
    {
        std::printf("**DEBUG** %s[%zd]: %g, %g\n", var_name.c_str(), i, r1[i], r2[i]);
    }

    auto predicate = [](double const a, double const b) {
        return std::abs(a - b) < std::numeric_limits<double>::epsilon();
    };
    if (!std::equal(r1.begin(), r1.end(), r2.begin(), predicate))
    {
        auto mism = std::mismatch(r1.begin(), r1.end(), r2.begin(), predicate);
        auto mism_index = std::distance(r1.begin(), mism.first);
        std::printf("different %s values at %zd: %g != %g.\n", var_name.c_str(),
            mism_index, *mism.first, *mism.second);
        return false;
    }
    std::printf("identical values of %s.\n", var_name.c_str());
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
            std::printf("different %s values at %zd.\n", var_name.c_str(), i);

            return false;
        }
    }
    std::printf("identical values of %s.\n", var_name.c_str());
    return true;
}

template <typename F>
bool check_run_result(fx_case test_case, F fx)
{
    fx_args& a = test_case.args;
    fx(a.opts_eos, a.opts_problem, a.physcon_A, a.physcon_B, a.physcon_c, a.er_i, a.fx_i, a.fy_i, a.fz_i,
        a.d, a.rho, a.sx, a.sy, a.sz, a.egas, a.tau, a.fgamma, a.U, a.mmw,
        a.X_spc, a.Z_spc, a.dt, a.clightinv);

    if (!are_ranges_same(test_case.args.egas, test_case.outs.egas, "egas"))
    {
        return false;
    }
    if (!are_ranges_same(test_case.args.sx, test_case.outs.sx, "sx"))
    {
        return false;
    }
    if (!are_ranges_same(test_case.args.sy, test_case.outs.sy, "sy"))
    {
        return false;
    }
    if (!are_ranges_same(test_case.args.sz, test_case.outs.sz, "sz"))
    {
        return false;
    }
    if (!are_ranges_same(test_case.args.U, test_case.outs.U, "U"))
    {
        return false;
    }

    return true;
}

bool check_case(size_t index)
{
    std::printf("***** load case %zd *****\n", index);
    fx_case const test_case = import_case(index);

    std::printf("***** cpu kernel (the original) *****\n");

    if (!check_run_result(test_case, radiation_cpu_kernel))
    {
        std::printf("code integrity check failed.\n");
        return false;
    }
    std::printf("code integrity check passed.\n");

    std::printf("***** gpu kernel (ported code) *****\n");
    if (!check_run_result(test_case, radiation_gpu_kernel))
    {
        std::printf("code integrity check failed.\n");
        return false;
    }
    std::printf("code integrity check passed.\n");

    return true;
}

int main()
{
    std::printf("epsilon: %g\n", std::numeric_limits<double>::epsilon());
    check_case(81);
    //for (size_t i = 0; i < 100; ++i)
    //{
    //    if (!check_case(i))
    //    {
    //        return 1;
    //    }
    //}

    return 0;
}
