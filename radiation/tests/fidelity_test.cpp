#include "kernels/kernel_cpu.hpp"
#include "kernels/kernel_v2.hpp"
#if OCTORAD_HAVE_CUDA
#include "kernels/kernel_gpu.hpp"
#endif
#include "utils/fx_case.hpp"
#include "utils/fx_compare.hpp"
#include "utils/util.hpp"

#include <cstdio>

constexpr std::size_t CASE_COUNT = OCTORAD_DUMP_COUNT;

template <typename K>
bool check_run_result(octotiger::fx_case test_case, K& kernel)
{
    octotiger::fx_args& a = test_case.args;
    kernel(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
        a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c, a.er_i,
        a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz, a.egas, a.tau,
        a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt, a.clightinv);

    bool result =
        octotiger::are_ranges_same(a.egas, test_case.outs.egas, "egas") &&
        octotiger::are_ranges_same(a.sx, test_case.outs.sx, "sx") &&
        octotiger::are_ranges_same(a.sy, test_case.outs.sy, "sy") &&
        octotiger::are_ranges_same(a.sz, test_case.outs.sz, "sz") &&
        octotiger::are_ranges_same(a.U, test_case.outs.U, "U");
    return result;
}

struct case_checker
{
    case_checker()
      : cpu_kernel()
      , v2_kernel()
      , gpu_kernel()
    {
    }
    case_checker(case_checker const& other) = delete;
    case_checker(case_checker&& other) = delete;

    bool operator()(std::size_t index)
    {
        std::printf("***** load case %zd *****\n", index);
        octotiger::fx_case const test_case = octotiger::import_case(index);

        std::printf("***** cpu kernel (reference) *****\n");

        if (!check_run_result(test_case, cpu_kernel))
        {
            std::printf("case %zd code integrity check failed.\n", index);
            return false;
        }
        std::printf("case %zd code integrity check passed.\n", index);

        std::printf("***** cpu kernel (v2) *****\n");

        if (!check_run_result(test_case, v2_kernel))
        {
            std::printf("case %zd code integrity check failed.\n", index);
            return false;
        }
        std::printf("case %zd code integrity check passed.\n", index);

#if OCTORAD_HAVE_CUDA
        std::printf("***** gpu kernel (ported code) *****\n");
        if (!check_run_result(test_case, gpu_kernel))
        {
            std::printf("case %zd code integrity check failed.\n", index);
            return false;
        }
        std::printf("case %zd code integrity check passed.\n", index);
#endif

        return true;
    }

private:
    octotiger::radiation_cpu_kernel cpu_kernel;
    octotiger::radiation_v2_kernel v2_kernel;
#if OCTORAD_HAVE_CUDA
    octotiger::radiation_gpu_kernel gpu_kernel;
#endif
};

int main()
{
    try
    {
        case_checker check_case;
        //check_case(78);
        for (std::size_t i = 0; i < CASE_COUNT / 100; ++i)
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
        return 1;
    }

    return 0;
}
