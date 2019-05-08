#include "utils/fx_case.hpp"
#include "utils/fx_compare.hpp"
#include "utils/util.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>

constexpr std::size_t const arr_size = 2744;

void investigate_array(std::vector<double> v, std::string const var_name)
{
    auto last = std::unique(v.begin(), v.end());
    auto distance = std::distance(v.begin(), last);
    std::printf(", %s: %3zd", var_name.c_str(), distance);
}

void investigate_case(std::size_t index)
{
    std::printf("case: %5zd", index);
    octotiger::fx_case const test_case = octotiger::import_case(index);
    bool same_size_arg_arrs = (arr_size == test_case.args.rho.size()) &&
        (arr_size == test_case.args.sx.size()) &&
        (arr_size == test_case.args.sy.size()) &&
        (arr_size == test_case.args.sz.size()) &&
        (arr_size == test_case.args.egas.size()) &&
        (arr_size == test_case.args.tau.size()) &&
        (arr_size == test_case.args.mmw.size()) &&
        (arr_size == test_case.args.X_spc.size()) &&
        (arr_size == test_case.args.Z_spc.size());
    for (auto const& sv : test_case.args.U)
    {
        same_size_arg_arrs = same_size_arg_arrs && sv.size();
    }
    if (!same_size_arg_arrs)
    {
        throw octotiger::formatted_exception(
            "not all arg arrays have the same % size", arr_size);
    }

    investigate_array(test_case.args.rho, "rho");
    investigate_array(test_case.args.sx, "sx");
    investigate_array(test_case.args.sy, "sy");
    investigate_array(test_case.args.sz, "sz");
    investigate_array(test_case.args.egas, "egas");
    investigate_array(test_case.args.tau, "tau");
    investigate_array(test_case.args.mmw, "mmw");
    investigate_array(test_case.args.X_spc, "X_spc");
    investigate_array(test_case.args.Z_spc, "Z_spc");
    for (std::size_t i = 0; i < test_case.args.U.size(); ++i)
    {
        investigate_array(test_case.args.U[i], octotiger::tprintf("U[%]", i));
    }
    std::printf("\n");
}

int main()
{
    try
    {
        //investigate_case(78);
        constexpr std::size_t case_count = 13140;
        for (std::size_t i = 0 ; i < case_count; i += 100)
        {
            investigate_case(i);
        }
    }
    catch (std::exception const& e)
    {
        std::printf("exception caught: %s\n", e.what());
        return 1;
    }

    return 0;
}
