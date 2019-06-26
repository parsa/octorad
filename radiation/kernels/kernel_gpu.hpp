#pragma once

#include "config.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

constexpr std::size_t RAD_GRID_I = RAD_NX - (2 * RAD_BW);
constexpr std::size_t GRID_ARRAY_SIZE = RAD_GRID_I * RAD_GRID_I * RAD_GRID_I;

namespace octotiger {
    namespace detail {

        struct output_payload_t
        {
            double sx[GRID_ARRAY_SIZE];
            double sy[GRID_ARRAY_SIZE];
            double sz[GRID_ARRAY_SIZE];
            double egas[GRID_ARRAY_SIZE];
            double tau[GRID_ARRAY_SIZE];
            double U[NRF][GRID_ARRAY_SIZE];
        };

        struct payload_t : output_payload_t
        {
            double rho[GRID_ARRAY_SIZE];
            double X_spc[GRID_ARRAY_SIZE];
            double Z_spc[GRID_ARRAY_SIZE];
            double mmw[GRID_ARRAY_SIZE];
        };

        struct device_payload_deleter
        {
            void operator()(payload_t* ptr);
        };

        struct host_payload_deleter
        {
            void operator()(payload_t* ptr);
        };

        using device_payload_ptr =
            std::unique_ptr<detail::payload_t, detail::device_payload_deleter>;

        using host_payload_ptr =
            std::unique_ptr<detail::payload_t, detail::host_payload_deleter>;
    }

    void device_init();
    void device_reset();

    struct radiation_gpu_kernel
    {
        radiation_gpu_kernel();
        radiation_gpu_kernel(radiation_gpu_kernel const&) = delete;
        radiation_gpu_kernel& operator=(radiation_gpu_kernel const&) = delete;
        radiation_gpu_kernel& operator=(radiation_gpu_kernel&&);
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
            std::array<std::vector<double>, NRF>& U,
            std::vector<double> const& mmw, std::vector<double> const& X_spc,
            std::vector<double> const& Z_spc, double const dt,
            double const clightinv);

    private:
        void load_args(detail::payload_t& h_payload, std::int64_t const d,
            std::vector<double>& sx, std::vector<double>& sy,
            std::vector<double>& sz, std::vector<double>& egas,
            std::vector<double>& tau, std::array<std::vector<double>, NRF>& U,
            std::vector<double> const& rho, std::vector<double> const& X_spc,
            std::vector<double> const& Z_spc, std::vector<double> const& mmw);
        void update_outputs(detail::payload_t& h_payload, std::int64_t const d,
            std::vector<double>& sx, std::vector<double>& sy,
            std::vector<double>& sz, std::vector<double>& egas,
            std::vector<double>& tau, std::array<std::vector<double>, NRF>& U);

        bool moved = false;
        detail::host_payload_ptr h_payload_ptr;
        std::size_t stream_index = static_cast<std::size_t>(-1);
    };
}
