#include "kernels/kernel_cpu.hpp"
#include "kernels/kernel_v2.hpp"
#define PROFILE_VARIANT_CPU_REF 0
#define PROFILE_VARIANT_V2 1
#define PROFILE_VARIANT_GPU 2

#ifndef PROFILE_VARIANT
#error "PROFILE_VARIANT must be defined"
#endif

#if PROFILE_VARIANT == PROFILE_VARIANT_GPU
#if !defined(OCTORAD_HAVE_CUDA)
#error "GPU profile requires CUDA support"
#endif
#include "kernels/kernel_gpu.hpp"
#endif
#include "utils/fx_case.hpp"
#include "utils/fx_compare.hpp"
#include "utils/scoped_timer.hpp"
#include "utils/util.hpp"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <ratio>
#include <vector>

#define LOAD_RANDOM_CASES 0

constexpr std::size_t CASE_COUNT = OCTORAD_DUMP_COUNT;

std::size_t choose_case_count()
{
    std::size_t limit = CASE_COUNT;
    if (char const* env = std::getenv("OCTORAD_PROFILE_CASES"))
    {
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(env, &end, 10);
        if (end != nullptr && end != env)
        {
            if (parsed < limit)
            {
                limit = static_cast<std::size_t>(parsed);
            }
        }
    }
    return limit == 0 ? 1 : limit;
}

#if PROFILE_VARIANT == PROFILE_VARIANT_CPU_REF
using selected_kernel = octotiger::radiation_cpu_kernel;
constexpr char const* k_profile_label = "CPU reference kernel";
constexpr bool k_use_device = false;
#elif PROFILE_VARIANT == PROFILE_VARIANT_V2
using selected_kernel = octotiger::radiation_v2_kernel;
constexpr char const* k_profile_label = "CPU v2 kernel";
constexpr bool k_use_device = false;
#elif PROFILE_VARIANT == PROFILE_VARIANT_GPU
using selected_kernel = octotiger::radiation_gpu_kernel;
constexpr char const* k_profile_label = "GPU kernel";
constexpr bool k_use_device = true;
#else
#    error "Unknown PROFILE_VARIANT"
#endif

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

#if PROFILE_VARIANT == PROFILE_VARIANT_GPU
double run_kernel_gpu_batched(std::vector<octotiger::fx_case>& test_cases)
{
    using octotiger::radiation_gpu_kernel;
    double overall_et{};
    double pure_et{};
    int num_streams = 4;
    if (char const* env = std::getenv("OCTORAD_PROFILE_STREAMS"))
    {
        int val = std::atoi(env);
        if (val > 0)
        {
            num_streams = val;
        }
    }
    std::vector<radiation_gpu_kernel> kernels(static_cast<std::size_t>(num_streams));
    {
        scoped_timer<double> timer(overall_et);
        {
            scoped_timer<double, std::micro> timer2(pure_et);
            for (std::size_t idx = 0; idx < test_cases.size(); ++idx)
            {
                auto& test_case = test_cases[idx];
                auto& a = test_case.args;
                auto& k = kernels[idx % kernels.size()];
                k.enqueue(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
                    a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c,
                    a.er_i, a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz,
                    a.egas, a.tau, a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt,
                    a.clightinv, /*copy_back*/ false);
            }
            for (auto& k : kernels)
            {
                k.stream_sync();
            }
        }
    }
    std::printf("total kernel execution time: %gus\n", pure_et);
    std::printf("overall execution time: %gs\n", overall_et);
    return pure_et;
}
#endif
static bool use_batched_mode()
{
    if (char const* env = std::getenv("OCTORAD_PROFILE_BATCHED"))
    {
        return std::atoi(env) != 0;
    }
    return false;
}
int main()
{
    try
    {
#if PROFILE_VARIANT == PROFILE_VARIANT_GPU
        std::printf("***** init device *****\n");
        double dev_init_et{};
        {
            scoped_timer<double, std::milli> timer(dev_init_et);
            octotiger::device_init();
        }
        std::printf("initialized device in %gms\n", dev_init_et);
#else
        std::printf("***** %s *****\n", k_profile_label);
#endif

        std::printf("***** load cases *****\n");
        std::vector<octotiger::fx_case> test_cases;
        double load_et{};
        std::size_t const cases_to_load =
            std::min<std::size_t>(CASE_COUNT, choose_case_count());
        {
            scoped_timer<double> timer(load_et);
            test_cases.reserve(cases_to_load);

            for (std::size_t i = 0; i < cases_to_load; ++i)
            {
                double const percent_loaded = 100.0 *
                    (static_cast<double>(i + 1) /
                        static_cast<double>(cases_to_load));
                std::printf("\rloaded %.3g%% of the cases\t", percent_loaded);
                std::fflush(stdout);
#if LOAD_RANDOM_CASES
                std::size_t case_id =
                    select_random_case(0, CASE_COUNT - 1);
                test_cases.emplace_back(octotiger::import_case(case_id));
#else
                test_cases.emplace_back(octotiger::import_case(i));
#endif
            }
        }
        std::printf(
            "\rloaded %zd cases in %gs\t\t\n", test_cases.size(), load_et);

        std::printf(">>> profiling %s on %zu cases\n", k_profile_label,
            test_cases.size());
        #if PROFILE_VARIANT == PROFILE_VARIANT_GPU
        if (use_batched_mode())
        {
            run_kernel_gpu_batched(test_cases);
        }
        else
        {
            run_kernel<selected_kernel>(test_cases);
        }
        #else
            run_kernel<selected_kernel>(test_cases);
        #endif
#if PROFILE_VARIANT == PROFILE_VARIANT_GPU
        octotiger::device_reset();
#endif
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        return 1;
    }

    return 0;
}
