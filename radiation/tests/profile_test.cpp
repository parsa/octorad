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
#include <vector>

#define LOAD_RANDOM_CASES 1
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

    void run_case_on_kernel(octotiger::fx_case test_case)
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

    double operator()(octotiger::fx_case const test_case)
    {
        double cpu_kernel_duration{};
        {
            scoped_timer<double>{cpu_kernel_duration};
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

int main()
{
    try
    {
        std::printf("***** init device *****\n");
        octotiger::device_init();

        std::printf("***** load cases *****\n");
        std::vector<octotiger::fx_case> test_cases;
        double load_et{};
        {
            scoped_timer<double, std::ratio<1>> timer(load_et);
            test_cases.reserve(LOAD_CASE_COUNT);

            for (std::size_t i = 0; i < LOAD_CASE_COUNT; ++i)
            {
                double const perecent_loaded = 100.0 * static_cast<double>(i) /
                    static_cast<double>(LOAD_CASE_COUNT);
                std::printf("\rloaded %g%% of the cases\t", perecent_loaded);
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

        double reference_execution_time{};
        double cpu_kernel_et{};
        {
            scoped_timer<double, std::ratio<1>> timer(cpu_kernel_et);
            std::printf("***** cpu kernel (reference) *****\n");
            double execution_time{};
            case_checker<octotiger::radiation_cpu_kernel> run_cpu_case(
                test_cases[0].data_size);
            for (auto& test_case : test_cases)
            {
                execution_time += run_cpu_case(test_case);
            }
            // CPU case is the reference execution time
            reference_execution_time = execution_time;

            std::printf("execution time: %gus\n", execution_time);
            std::printf("speedup: %g\n", 1.0);
        }
        std::printf("duration: %gs\n", cpu_kernel_et);

        double v2_kernel_et{};
        {
            scoped_timer<double, std::ratio<1>> timer(v2_kernel_et);
            std::printf("***** cpu kernel (v2) *****\n");
            double execution_time{};
            case_checker<octotiger::radiation_v2_kernel> run_v2_case(
                test_cases[0].data_size);
            for (auto& test_case : test_cases)
            {
                execution_time += run_v2_case(test_case);
            }
            std::printf("execution time: %gus\n", execution_time);
            std::printf(
                "speedup: %g\n", reference_execution_time / execution_time);
        }
        std::printf("duration: %gs\n", v2_kernel_et);

#if OCTORAD_HAVE_CUDA
        double gpu_kernel_et{};
        {
            scoped_timer<double, std::ratio<1>> timer(gpu_kernel_et);
            std::printf("***** gpu kernel (ported code) *****\n");
            double execution_time{};
            case_checker<octotiger::radiation_gpu_kernel> run_gpu_case(
                test_cases[0].data_size);
            for (auto& test_case : test_cases)
            {
                execution_time += run_gpu_case(test_case);
            }
            std::printf("execution time: %gus\n", execution_time);
            std::printf(
                "speedup: %g\n", reference_execution_time / execution_time);
        }
        std::printf("duration: %gs\n", gpu_kernel_et);
#endif
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        return 1;
    }

    return 0;
}
