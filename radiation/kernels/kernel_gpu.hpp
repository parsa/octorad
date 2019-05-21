#pragma once

#include "config.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace octotiger {
    void device_init();

    struct radiation_gpu_kernel
    {
        radiation_gpu_kernel();
        radiation_gpu_kernel(radiation_gpu_kernel const&) = delete;
        radiation_gpu_kernel(radiation_gpu_kernel&& other);
        ~radiation_gpu_kernel();
        void operator()(std::int64_t const opts_eos,
            std::int64_t const opts_problem, double const opts_dual_energy_sw1,
            double const opts_dual_energy_sw2, double const physcon_A,
            double const physcon_B, double const physcon_c,
            std::int64_t const er_i, std::int64_t const fx_i,
            std::int64_t const fy_i, std::int64_t const fz_i,
            std::int64_t const d, std::vector<double> const& rho,
            std::vector<double>& sx, std::vector<double>& sy,
            std::vector<double>& sz, std::vector<double>& egas,
            std::vector<double>& tau, double const fgamma,
            std::array<std::vector<double>, NRF> U,
            std::vector<double> const mmw, std::vector<double> const X_spc,
            std::vector<double> const Z_spc, double const dt,
            double const clightinv);

    private:
        bool moved = false;
        double* d_payload = nullptr;
        double* h_payload_ptr = nullptr;
        std::size_t stream_index;
    };
}
