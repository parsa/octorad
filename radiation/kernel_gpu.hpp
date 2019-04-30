#pragma once

#include "config.hpp"

#include <array>
#include <vector>

void radiation_gpu_kernel(std::int64_t const er_i, std::int64_t const fx_i,
    std::int64_t const fy_i, std::int64_t const fz_i, std::int64_t const d,
    std::vector<double> const& rho, std::vector<double>& sx,
    std::vector<double>& sy, std::vector<double>& sz, std::vector<double>& egas,
    std::vector<double>& tau, double const fgamma,
    std::array<std::vector<double>, NRF> U, std::vector<double> const mmw,
    std::vector<double> const X_spc, std::vector<double> const Z_spc,
    double const dt, double const clightinv);
