#include "kernels/kernel_gpu.hpp"
#include "utils/fx_case.hpp"
#include "utils/fx_compare.hpp"
#include "utils/util.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <ratio>
#include <vector>

namespace ot = octotiger;

template <typename K>
void check_case(K& kernel, ot::fx_case test_case)
{
    ot::fx_args& a = test_case.args;
    kernel(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
        a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c, a.er_i,
        a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz, a.egas, a.tau,
        a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt, a.clightinv);

    bool const success =    //
        ot::are_ranges_same(a.egas, test_case.outs.egas, "egas") &&
        ot::are_ranges_same(a.sx, test_case.outs.sx, "sx") &&
        ot::are_ranges_same(a.sy, test_case.outs.sy, "sy") &&
        ot::are_ranges_same(a.sz, test_case.outs.sz, "sz") &&
        ot::are_ranges_same(a.U, test_case.outs.U, "U");
    if (!success)
    {
        throw ot::formatted_exception(
            "case % code integrity check failed", test_case.index);
    }
}

int main()
{
    try
    {
        ot::device_init();

        ot::fx_case test_case = ot::import_case(83);

        ot::radiation_gpu_kernel kernel;
        check_case(kernel, test_case);
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        return 1;
    }

    return 0;
}
