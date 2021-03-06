#include "kernels/kernel_gpu.hpp"
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

constexpr std::size_t ITERATIONS = 13'139;
constexpr std::size_t MAX_STREAMS = 4;

template <typename K>
struct case_runner
{
    double operator()(octotiger::fx_case test_case)
    {
        double duration = 0.0;
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
};

template <typename K>
void profile_kernel(octotiger::fx_case& test_case)
{
    double overall_et = 0.;
    double pure_et = 0.;
    {
        scoped_timer<double> timer(overall_et);

        // MAX_STREAMS kernels, each with a different stream
        std::vector<case_runner<K>> workers(MAX_STREAMS);

        // initial MAX_STREAMS cases
        std::vector<std::future<double>> case_queue;
        case_queue.reserve(MAX_STREAMS);
        for (std::size_t wi = 0; wi < MAX_STREAMS; ++wi)
        {
            case_queue.emplace_back(std::async(std::launch::async,
                [&test_case, &w = workers[wi]]() { return w(test_case); }));
        }

        // MAX_STREAM initial values are already in the queue
        std::size_t case_queue_top = MAX_STREAMS;
        // rest of the cases until queue is empty
        while (case_queue_top < ITERATIONS)
        {
            // give new work to slots with finished tasks
            for (std::size_t strm_idx = 0; strm_idx < case_queue.size();
                 ++strm_idx)
            {
                std::future<double>& f = case_queue[strm_idx];
                // is future ready?
                if (std::future_status::ready ==
                    f.wait_for(std::chrono::seconds::zero()))
                {
                    // get value out of the future
                    pure_et += f.get();
                    // schedule new task
                    f = std::async(std::launch::async,
                        [&test_case, &w = workers[strm_idx]]() {
                            return w(test_case);
                        });
                    ++case_queue_top;
                }
            }
        }
        // remaining MAX_STREAMS
        for (std::future<double>& t : case_queue)
        {
            pure_et += t.get();
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

        octotiger::device_reset();
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        return 1;
    }

    return 0;
}
