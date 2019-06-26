#include "config.hpp"
#include "utils/fx_compare.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace octotiger {
    // NOTE: possible room for improvement: https://stackoverflow.com/q/17333
    bool almost_equal(double const a, double const b)
    {
        double const diff = std::abs(a - b);
        //double const largest = std::max(std::abs(a), std::abs(b));
        //return diff < largest * std::numeric_limits<double>::epsilon();
        //return diff < 1e-5;
        return diff <= std::numeric_limits<double>::epsilon();
    }

    bool are_ranges_same(std::vector<double> const& r1,
        std::vector<double> const& r2, std::string const& var_name)
    {
        if (!std::equal(r1.begin(), r1.end(), r2.begin(), almost_equal))
        {
            auto mism =
                std::mismatch(r1.begin(), r1.end(), r2.begin(), almost_equal);
            auto mism_index = std::distance(r1.begin(), mism.first);

            std::printf("mismatch in %s[%zd]: %g != %g.\n", var_name.c_str(),
                mism_index, *mism.first, *mism.second);

            size_t mism_count = 0;
            for (; mism.first != r1.end(); ++mism_count)
            {
                mism = std::mismatch(
                    ++mism.first, r1.end(), ++mism.second, almost_equal);
            }
            std::printf(
                "mismatch count in %s: %zd.\n", var_name.c_str(), mism_count);
            return false;
        }
        return true;
    }

    bool are_ranges_same(std::array<std::vector<double>, NRF> const& r1,
        std::array<std::vector<double>, NRF> const& r2,
        std::string const& var_name)
    {
        for (std::size_t i = 0; i < NRF; ++i)
        {
            std::string var_name_i = var_name + "[" + std::to_string(i) + "]";
            if (!are_ranges_same(r1[i], r2[i], var_name_i))
            {
                return false;
            }
        }
        return true;
    }
}
