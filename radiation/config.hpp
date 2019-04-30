#pragma once

#include <cstddef>
#include <cstdint>

constexpr size_t NRF = 4;
constexpr std::size_t NDIM = 3;
constexpr double MARSHAK_OPAC = 1.0e+2;

constexpr std::int64_t INX = 8;

constexpr std::int64_t RAD_BW = 3;
constexpr std::int64_t RAD_NX = INX + 2 * RAD_BW;
constexpr std::int64_t RAD_N3 = RAD_NX * RAD_NX * RAD_NX;

constexpr std::int64_t H_BW = 3;
constexpr std::int64_t H_NX = 2 * H_BW + INX;

constexpr std::int64_t H_DNX = H_NX * H_NX;
constexpr std::int64_t H_DNY = H_NX;
constexpr std::int64_t H_DNZ = 1;

enum class eos
{
    ideal,
    wd
};

extern eos opts_eos;

constexpr double de_switch1 = 0.10000000000000001;
constexpr double de_switch2 = 0.0010000000000000000;
