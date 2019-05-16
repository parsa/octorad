#include "kernels/kernel_gpu.hpp"
#include "utils/fx_case.hpp"
#include "utils/fx_compare.hpp"
#include "utils/util.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <ratio>
#include <vector>

#define VERIFY_OUTCOMES 0

constexpr std::size_t CASE_COUNT = OCTORAD_DUMP_COUNT;
constexpr std::size_t LOAD_CASE_COUNT = 100;

template <typename K>
struct case_checker
{
    case_checker(std::size_t data_size)
      : kernel(data_size)
    {
    }
    case_checker(case_checker const& other) = delete;
    case_checker(case_checker&& other) = delete;

    void operator()(octotiger::fx_case test_case)
    {
        using octotiger::are_ranges_same;

        octotiger::fx_args& a = test_case.args;
        kernel(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
            a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c,
            a.er_i, a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz,
            a.egas, a.tau, a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt,
            a.clightinv);

#if VERIFY_OUTCOMES
        if (verify_outcome)
        {
            bool const success =
                are_ranges_same(test_case.args.egas, test_case.outs.egas, "egas") &&
                are_ranges_same(test_case.args.sx, test_case.outs.sx, "sx") &&
                are_ranges_same(test_case.args.sy, test_case.outs.sy, "sy") &&
                are_ranges_same(test_case.args.sz, test_case.outs.sz, "sz") &&
                are_ranges_same(test_case.args.U, test_case.outs.U, "U");
            if (!success)
            {
                throw octotiger::formatted_exception(
                    "case % code integrity check failed", test_case.index);
            }
        }
#endif
    }

private:
    K kernel;
};

int main()
{
    try
    {
        std::printf("***** init device *****\n");
        octotiger::device_init();

        std::printf("***** load case *****\n");
        octotiger::fx_case test_case = octotiger::import_case(83);

        std::printf("***** gpu kernel (ported code) *****\n");
        case_checker<octotiger::radiation_gpu_kernel> run_cpu_case(
            test_case.data_size);
        run_cpu_case(test_case);
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        return 1;
    }

    return 0;
}
