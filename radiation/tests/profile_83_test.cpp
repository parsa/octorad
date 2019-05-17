#include "kernels/kernel_cpu.hpp"
#include "kernels/kernel_v2.hpp"
#if OCTORAD_HAVE_CUDA
#include "kernels/kernel_gpu.hpp"
#endif
#include "utils/fx_case.hpp"
#include "utils/fx_compare.hpp"
#include "utils/scoped_timer.hpp"
#include "utils/util.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <ratio>
#include <vector>

constexpr std::size_t ITERATIONS = 10000;

template <typename K>
struct case_runner
{
    double operator()(octotiger::fx_case test_case)
    {
        duration = 0.0;
        {
            scoped_timer<double, std::micro>{duration};
            octotiger::fx_args& a = test_case.args;
            kernel(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
                a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c,
                a.er_i, a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz,
                a.egas, a.tau, a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt,
                a.clightinv);
        }
        return duration;
    }

private:
    K kernel;
    double duration{};
};

template <typename K>
void profile_kernel(octotiger::fx_case& test_case)
{
    double overall_et{};
    double pure_et{};
    {
        scoped_timer<double> timer(overall_et);
        case_runner<K> run_cpu_case;
        for (std::size_t i = 0; i < ITERATIONS; ++i)
        {
            pure_et += run_cpu_case(test_case);
        }
    }
    std::printf("total kernel execution time: %gus\n", pure_et);
    std::printf("overall execution time: %gs\n", overall_et);
}

int main()
{
    try
    {
        octotiger::device_init();

        octotiger::fx_case test_case = octotiger::import_case(83);

        profile_kernel<octotiger::radiation_gpu_kernel>(test_case);
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        return 1;
    }

    return 0;
}