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

int main()
{
    try
    {
        ot::device_init();

        ot::fx_case test_case = ot::import_case(83);
        ot::fx_args& a = test_case.args;

        ot::radiation_gpu_kernel kernel;

        kernel(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
            a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c,
            a.er_i, a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz,
            a.egas, a.tau, a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt,
            a.clightinv);
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        return 1;
    }

    return 0;
}
