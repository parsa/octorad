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
#include <future>
#include <random>
#include <ratio>
#include <utility>
#include <vector>

constexpr std::size_t ITERATIONS = 10000;

using run_ret_t = std::pair<double, std::size_t>;

template <typename K>
struct case_runner
{
    case_runner(std::size_t idx)
    {
        index = idx;
    }
    run_ret_t operator()(octotiger::fx_case test_case)
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
        return std::make_pair(duration, index);
    }

private:
    K kernel;
    double duration{};
    std::size_t index;
};

template <typename K>
void profile_kernel(octotiger::fx_case& test_case)
{
    double overall_et{};
    double pure_et{};
    {
        scoped_timer<double> timer(overall_et);

        //// 3 kernels, each with a different stream
        std::vector<case_runner<K>> workers;
        for(std::size_t wi = 0; wi < 3; ++wi)
        {
            workers.emplace_back(case_runner<K>{wi});
        }

        // initial 3 cases
        std::vector<std::future<run_ret_t>> case_queue;
        //std::size_t idx = 0;
        //for (std::size_t wi = 0; wi < 3; wi++)
        //{
        //    case_queue.emplace_back(std::async(std::launch::async,
        //        [&]() { return workers[wi](test_case); }));
        //    ++idx;
        //}
        //// rest of the cases
        //while (idx < ITERATIONS)
        //{
        //    for (auto& t : case_queue)
        //    {
        //        if (std::future_status::ready ==
        //            t.wait_for(std::chrono::seconds::zero()))
        //        {
        //            auto r = t.get();
        //            pure_et += r.first;
        //            t = std::async(std::launch::async, [&, i = r.second]() {
        //                return workers[i](test_case);
        //            });
        //            ++idx;
        //        }
        //    }
        //}

        auto f0 = std::async(
            std::launch::async, [&]() { return workers[0](test_case); });
        auto f1 = std::async(
            std::launch::async, [&]() { return workers[1](test_case); });
        //auto f2 = std::async(
        //    std::launch::async, [&]() { return workers[2](test_case); });

        auto ff0 = std::move(case_queue[0]);
        auto ff1 = std::move(case_queue[1]);
        auto ff2 = std::move(case_queue[2]);

        ff0.get();
        ff1.get();
        ff2.get();

        //for (std::size_t wi = 0; wi < 3; ++wi)
        //{
        //    case_queue.emplace_back(std::async(
        //        std::launch::async, [&]() { return workers[2](test_case); }));
        //}

        //case_queue[0].get();
        //case_queue[1].get();
        //case_queue[2].get();
        //pure_et += case_queue[0].get().first;
        //pure_et += case_queue[1].get().first;
        //pure_et += case_queue[2].get().first;

        //case_runner<K> run_kernel_case;
        //for (std::size_t i = 0; i < ITERATIONS; ++i)
        //{
        //    pure_et += run_kernel_case(test_case);
        //}
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
