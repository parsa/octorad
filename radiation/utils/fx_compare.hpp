#pragma once

#include "config.hpp"

#include <array>
#include <string>
#include <vector>

namespace octotiger {
    bool almost_equal(double const a, double const b);
    bool are_ranges_same(std::vector<double> const& r1,
        std::vector<double> const& r2, std::string const var_name);
    bool are_ranges_same(std::array<std::vector<double>, NRF> const& r1,
        std::array<std::vector<double>, NRF> const& r2,
        std::string const var_name);
}
