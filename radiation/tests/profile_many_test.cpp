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

#define LOAD_RANDOM_CASES 0

constexpr std::size_t CASE_COUNT = OCTORAD_DUMP_COUNT;
constexpr std::size_t LOAD_CASE_COUNT = 13000;

template <typename K>
struct case_runner
{
    case_runner()
      : kernel()
    {
    }
    case_runner(case_runner const& other) = delete;
    case_runner(case_runner&& other) = delete;

    void run_case_on_kernel(octotiger::fx_case test_case)
    {
        using octotiger::are_ranges_same;

        octotiger::fx_args& a = test_case.args;
        kernel(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
            a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c,
            a.er_i, a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz,
            a.egas, a.tau, a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt,
            a.clightinv);
    }

    double operator()(octotiger::fx_case const test_case)
    {
        double cpu_kernel_duration{};
        {
            scoped_timer<double, std::micro>{cpu_kernel_duration};
            run_case_on_kernel(test_case);
        }
        return cpu_kernel_duration;
    }

private:
    K kernel;
};

#if LOAD_RANDOM_CASES
std::size_t select_random_case(std::size_t min_val, std::size_t max_val)
{
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_int_distribution<std::size_t> dist(min_val, max_val);

    return dist(mt);
}
#endif

template <typename K>
double run_kernel(std::vector<octotiger::fx_case>& test_cases)
{
    double overall_et{};
    double pure_et{};
    {
        scoped_timer<double> timer(overall_et);
        case_runner<K> run_cpu_case;
        for (auto& test_case : test_cases)
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

        std::printf("***** load cases *****\n");
        std::vector<octotiger::fx_case> test_cases;
        double load_et{};
        {
            scoped_timer<double> timer(load_et);
            test_cases.reserve(LOAD_CASE_COUNT);

            for (std::size_t i = 0; i < LOAD_CASE_COUNT; ++i)
            {
                double const perecent_loaded = 100.0 * static_cast<double>(i) /
                    static_cast<double>(LOAD_CASE_COUNT);
                std::printf("\rloaded %.3g%% of the cases\t", perecent_loaded);
                std::fflush(stdout);
#if LOAD_RANDOM_CASES
                std::size_t case_id =
                    select_random_case(0, LOAD_CASE_COUNT - 1);
                test_cases.emplace_back(octotiger::import_case(case_id));
#else
                test_cases.emplace_back(octotiger::import_case(i));
#endif
            }
        }
        std::printf(
            "\rloaded %zd cases in %gs\t\t\n", test_cases.size(), load_et);

        auto gpu_k_et = run_kernel<octotiger::radiation_gpu_kernel>(test_cases);
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        return 1;
    }

    return 0;
}
