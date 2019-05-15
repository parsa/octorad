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

constexpr std::size_t case_count = OCTORAD_DUMP_COUNT;
constexpr std::size_t load_case_count = 100;

template <typename K>
struct case_checker
{
    case_checker(std::size_t data_size)
      : kernel(data_size)
    {
    }
    case_checker(case_checker const& other) = delete;
    case_checker(case_checker&& other) = delete;

    void run_case_on_kernel(octotiger::fx_case test_case, K kernel,
        bool const verify_outcome = false)
    {
        using octotiger::are_ranges_same;

        octotiger::fx_args& a = test_case.args;
        kernel(a.opts_eos, a.opts_problem, a.opts_dual_energy_sw1,
            a.opts_dual_energy_sw2, a.physcon_A, a.physcon_B, a.physcon_c,
            a.er_i, a.fx_i, a.fy_i, a.fz_i, a.d, a.rho, a.sx, a.sy, a.sz,
            a.egas, a.tau, a.fgamma, a.U, a.mmw, a.X_spc, a.Z_spc, a.dt,
            a.clightinv);

        //if (verify_outcome)
        //{
        //    bool const success =
        //        are_ranges_same(test_case.args.egas, test_case.outs.egas, "egas") &&
        //        are_ranges_same(test_case.args.sx, test_case.outs.sx, "sx") &&
        //        are_ranges_same(test_case.args.sy, test_case.outs.sy, "sy") &&
        //        are_ranges_same(test_case.args.sz, test_case.outs.sz, "sz") &&
        //        are_ranges_same(test_case.args.U, test_case.outs.U, "U");
        //    if (!success)
        //    {
        //        throw octotiger::formatted_exception(
        //            "case % code integrity check failed", test_case.index);
        //    }
        //}
    }

    double operator()(octotiger::fx_case const test_case)
    {
        double cpu_kernel_duration{};
        {
            K krnl(test_case.data_size);
            scoped_timer<double>{cpu_kernel_duration};
            run_case_on_kernel(test_case, std::move(krnl));
        }
        return cpu_kernel_duration;
    }

private:
    K kernel;
};

std::size_t select_random_case()
{
    static std::random_device rd;
    static std::mt19937 mt(rd());
    static std::uniform_int_distribution<std::size_t> dist(0, case_count);

    return dist(mt);
}

int main()
{
    try
    {
        std::printf("***** load cases *****\n");
        std::vector<octotiger::fx_case> test_cases;
        test_cases.reserve(100);

        for (std::size_t i = 0; i < load_case_count; ++i)
        {
            std::size_t case_id = select_random_case();
            std::printf("\rloading case %zd", case_id);
            //test_cases.emplace_back(octotiger::import_case(i));
            test_cases.emplace_back(
                octotiger::import_case(case_id));
        }
        std::printf("\rloaded %zd cases     \n", load_case_count);

        double reference_execution_time{};
        {
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

            std::printf("execution time: %gns\n", execution_time);
            std::printf("speedup: %g\n", 1.0);
        }

        {
            std::printf("***** cpu kernel (v2) *****\n");
            double execution_time{};
            case_checker<octotiger::radiation_v2_kernel> run_v2_case(
                test_cases[0].data_size);
            for (auto& test_case : test_cases)
            {
                execution_time += run_v2_case(test_case);
            }
            std::printf("execution time: %gns\n", execution_time);
            std::printf(
                "speedup: %g\n", reference_execution_time / execution_time);
        }

#if OCTORAD_HAVE_CUDA
        {
            std::printf("***** gpu kernel (ported code) *****\n");
            double execution_time{};
            case_checker<octotiger::radiation_gpu_kernel> run_gpu_case(
                test_cases[0].data_size);
            for (auto& test_case : test_cases)
            {
                execution_time += run_gpu_case(test_case);
            }
            std::printf("execution time: %gns\n", execution_time);
            std::printf(
                "speedup: %g\n", reference_execution_time / execution_time);
        }
#endif
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        return 1;
    }

    return 0;
}
