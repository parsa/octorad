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

        constexpr std::size_t MAX_STREAMS = 4;

        // MAX_STREAMS kernels, each with a different stream
        std::vector<case_runner<K>> workers;
        workers.reserve(MAX_STREAMS);
        for (std::size_t wi = 0; wi < MAX_STREAMS; ++wi)
        {
            workers.emplace_back(case_runner<K>{wi});
        }

        // initial MAX_STREAMS cases
        std::vector<std::future<run_ret_t>> case_queue;
        case_queue.reserve(MAX_STREAMS);
        for (std::size_t wi = 0; wi < MAX_STREAMS; ++wi)
        {
            case_queue.emplace_back(std::async(std::launch::async,
                [&test_case, &w = workers[wi]]() { return w(test_case); }));
        }

        // MAX_STREAM initial values are already in the queue
        std::size_t idx = MAX_STREAMS;
        // rest of the cases until queue is empty
        while (idx < ITERATIONS)
        {
            // give new work to slots with finished tasks
            for (auto& t : case_queue)
            {
                // is future ready?
                if (std::future_status::ready ==
                    t.wait_for(std::chrono::seconds::zero()))
                {
                    // get value out of the future
                    run_ret_t const res = t.get();
                    pure_et += res.first;
                    std::size_t const strm_idx = res.second;
                    // schedule new task
                    t = std::async(std::launch::async,
                        [&test_case, &w = workers[strm_idx]]() {
                            return w(test_case);
                        });
                    ++idx;
                }
            }
        }
        // remaining MAX_STREAMS
        for (auto& t : case_queue)
        {
            run_ret_t const res = t.get();
            pure_et += res.first;
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
