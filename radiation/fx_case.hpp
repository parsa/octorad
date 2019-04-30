#pragma once

#include "config.hpp"

#include <array>
#include <cstdint>
#include <vector>

struct fx_args
{
    std::int64_t opts_eos;
    std::int64_t opts_problem;
    double opts_dual_energy_sw1;
    double opts_dual_energy_sw2;
    double physcon_A;
    double physcon_B;
    double physcon_c;
    std::int64_t er_i{};
    std::int64_t fx_i{};
    std::int64_t fy_i{};
    std::int64_t fz_i{};
    std::int64_t d{};
    std::vector<double> rho{};
    std::vector<double> sx{};
    std::vector<double> sy{};
    std::vector<double> sz{};
    std::vector<double> egas{};
    std::vector<double> tau{};
    double fgamma{};
    std::array<std::vector<double>, NRF> U{};
    std::vector<double> mmw{};
    std::vector<double> X_spc{};
    std::vector<double> Z_spc{};
    double dt{};
    double clightinv{};
};

struct fx_outs
{
    std::vector<double> sx{};
    std::vector<double> sy{};
    std::vector<double> sz{};
    std::vector<double> egas{};
    std::array<std::vector<double>, NRF> U{};
};

struct fx_case
{
    fx_args args;
    fx_outs outs;
};

fx_case import_case(std::size_t index);
