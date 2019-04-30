#include "config.hpp"
#include "fx_case.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

constexpr char const* const basepath = "C:\\Users\\Parsa\\Desktop\\arg-dumps\\octotiger-radiation-";

std::vector<double> load_v(std::istream& is, std::string var_name)
{
    std::size_t size{};
    is.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));

    std::vector<double> v(size);
    is.read(reinterpret_cast<char*>(&v[0]), size * sizeof(double));

    std::printf("loaded vector<double>{%zd} %s.\n", size, var_name.c_str());

    return v;
}

std::array<std::vector<double>, NRF> load_a(std::istream& is, std::string var_name)
{
    std::size_t size{};
    is.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));
    if (size != NRF)
    {
        std::abort();
    }

    std::array<std::vector<double>, NRF> a{};
    for (auto& e : a)
    {
        std::string var_name_i = var_name + "[size_t]";
        e = load_v(is, var_name_i);
    }

    std::printf("loaded array<vector<size_t>, %zd> %s.\n", size, var_name.c_str());

    return a;
}

std::int64_t load_i(std::istream& is, std::string var_name)
{
    std::int64_t i{};
    is.read(reinterpret_cast<char*>(&i), sizeof(std::int64_t));

    std::printf("loaded int64_t %s.\n", var_name.c_str());

    return i;
}

double load_d(std::istream& is, std::string var_name)
{
    double d{};
    is.read(reinterpret_cast<char*>(&d), sizeof(double));

    std::printf("loaded double %s.\n", var_name.c_str());

    return d;
}

fx_args load_case_args(std::size_t index)
{
    std::string args_fn =
        std::string{basepath} + std::to_string(index) + std::string{".args"};
    std::ifstream is{args_fn, std::ios::binary};

    if (!is)
    {
        std::printf("cannot open file \"%s\".", args_fn.c_str());
        std::abort();
    }

    fx_args args;
    args.er_i = load_i(is, "er_i");
    args.fx_i = load_i(is, "fx_i");
    args.fy_i = load_i(is, "fy_i");
    args.fz_i = load_i(is, "fz_i");
    args.d = load_i(is, "d");
    args.rho = std::move(load_v(is, "rho"));
    args.sx = std::move(load_v(is, "sx"));
    args.sy = std::move(load_v(is, "sy"));
    args.sz = std::move(load_v(is, "sz"));
    args.egas = std::move(load_v(is, "egas"));
    args.tau = std::move(load_v(is, "tau"));
    args.fgamma = load_d(is, "fgamma");
    args.U = std::move(load_a(is, "U"));
    args.mmw = std::move(load_v(is, "mmw"));
    args.X_spc = std::move(load_v(is, "X_spc"));
    args.Z_spc = std::move(load_v(is, "Z_spc"));
    args.dt = load_d(is, "dt");
    args.clightinv = load_d(is, "clightinv");

    std::printf("loaded case args.\n");

    return args;
}

fx_outs load_case_outs(std::size_t index)
{
    std::string outs_fn =
        std::string{basepath} + std::to_string(index) + std::string{".outs"};
    std::ifstream is{outs_fn, std::ios::binary};

    if (!is)
    {
        std::printf("cannot open file \"%s\".", outs_fn.c_str());
        std::abort();
    }

    fx_outs outs;
    outs.sx = std::move(load_v(is, "sx"));
    outs.sy = std::move(load_v(is, "sy"));
    outs.sz = std::move(load_v(is, "sz"));
    outs.egas = std::move(load_v(is, "egas"));
    outs.U = std::move(load_a(is, "U"));

    std::printf("loaded case outputs\n");

    return outs;
}

fx_case import_case(std::size_t index)
{
    fx_case ret;

    ret.args = std::move(load_case_args(index));
    ret.outs = std::move(load_case_outs(index));

    std::printf("loaded case %zd.\n", index);
    return ret;
}
