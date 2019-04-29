#include "config.hpp"
#include "fx_case.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

constexpr char const* const basepath = "C:\\Users\\Parsa\\Desktop\\arg-dumps\\octotiger-radiation-";

std::vector<double> load_v(std::istream& is)
{
    std::size_t size{};
    is.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));

    std::vector<double> v(size);
    is.read(reinterpret_cast<char*>(&v[0]), size * sizeof(double));

    std::printf("vector<double>{%zd} read\n", size);

    return v;
}

std::array<std::vector<double>, NRF> load_a(std::istream& is)
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
        e = load_v(is);
    }

    std::printf("array<vector<size_t>, %zd> read\n", size);

    return a;
}

std::int64_t load_i(std::istream& is)
{
    std::int64_t i{};
    is.read(reinterpret_cast<char*>(&i), sizeof(std::int64_t));

    std::printf("std::int64_t read\n");

    return i;
}

double load_d(std::istream& is)
{
    double d{};
    is.read(reinterpret_cast<char*>(&d), sizeof(double));

    std::printf("double read\n");

    return d;
}

fx_args load_case_args(std::size_t index)
{
    std::string args_fn =
        std::string{basepath} + std::to_string(index) + std::string{".args"};
    std::ifstream is{args_fn, std::ios::binary};

    if (!is)
    {
        std::printf("cannot open file \"%s\"", args_fn.c_str());
        std::abort();
    }

    fx_args args;
    args.er_i = load_i(is);
    args.fx_i = load_i(is);
    args.fy_i = load_i(is);
    args.fz_i = load_i(is);
    args.d = load_i(is);
    args.rho = std::move(load_v(is));
    args.sx = std::move(load_v(is));
    args.sy = std::move(load_v(is));
    args.sz = std::move(load_v(is));
    args.egas = std::move(load_v(is));
    args.tau = std::move(load_v(is));
    args.fgamma = load_d(is);
    args.U = std::move(load_a(is));
    args.mmw = std::move(load_v(is));
    args.X_spc = std::move(load_v(is));
    args.Z_spc = std::move(load_v(is));
    args.dt = load_d(is);
    args.clightinv = load_d(is);

    std::printf("case args read\n");

    return args;
}

fx_outs load_case_outs(std::size_t index)
{
    std::string outs_fn =
        std::string{basepath} + std::to_string(index) + std::string{".outs"};
    std::ifstream is{outs_fn, std::ios::binary};

    if (!is)
    {
        std::printf("cannot open file \"%s\"", outs_fn.c_str());
        std::abort();
    }

    fx_outs outs;
    outs.sx = std::move(load_v(is));
    outs.sy = std::move(load_v(is));
    outs.sz = std::move(load_v(is));
    outs.egas = std::move(load_v(is));
    outs.U = std::move(load_a(is));

    std::printf("case outs read\n");

    return outs;
}

fx_case import_case(std::size_t index)
{
    fx_case ret;

    ret.args = std::move(load_case_args(index));
    ret.outs = std::move(load_case_outs(index));

    std::printf("case %zd read\n", index);
    return ret;
}
