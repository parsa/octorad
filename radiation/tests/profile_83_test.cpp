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
        double cpu_kernel_duration{};
        {
            scoped_timer<double, std::micro>{cpu_kernel_duration};
            octotiger::fx_args& a = test_case.args;
            kernel(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
                a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c,
                a.er_i, a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz,
                a.egas, a.tau, a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt,
                a.clightinv);
        }
        return cpu_kernel_duration;
    }

private:
    K kernel;
};

template <typename K>
double profile_kernel(octotiger::fx_case& test_case)
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

    return pure_et;
}

int main()
{
    try
    {
        std::printf("***** init device *****\n");
        double dev_init_et{};
        {
            scoped_timer<double, std::milli> timer(dev_init_et);
            octotiger::device_init();
        }
        std::printf("initialized device in %gms\n", dev_init_et);

        std::printf("***** load case 83 *****\n");
        octotiger::fx_case test_case;
        double load_et{};
        {
            scoped_timer<double> timer(load_et);

            test_case = octotiger::import_case(83);
        }
        std::printf("loaded case 83 in %gs\n", load_et);

        auto gpu_k_et =
            profile_kernel<octotiger::radiation_gpu_kernel>(test_case);
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        return 1;
    }

    return 0;
}
