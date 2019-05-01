#include "config.hpp"
#include "fx_case.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

constexpr char const* const basepath = OCTORAD_DUMP_DIR "/octotiger-radiation-";

std::vector<double> load_v(std::istream& is, std::string const var_name)
{
    std::streampos orig_pos{};
    std::streampos actual_pos = is.tellg();
    is.read(reinterpret_cast<char*>(&orig_pos), sizeof(std::streampos));

    if (actual_pos!= orig_pos)
    {
        std::printf("error: stream positions do not match. actual: %zd, "
                    "original: %zd\n",
            static_cast<std::size_t>(actual_pos),
            static_cast<std::size_t>(orig_pos));
        std::abort();
    }
    std::printf("matched stream positions: %zd\n",
        static_cast<std::size_t>(actual_pos));

    std::size_t size{};
    is.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));

    std::vector<double> v(size);
    is.read(reinterpret_cast<char*>(&v[0]), size * sizeof(double));

    std::printf("loaded vector<double>{%zd} %s.\n", size, var_name.c_str());

    return v;
}

std::array<std::vector<double>, NRF> load_a(std::istream& is, std::string const var_name)
{
    std::size_t size{};
    is.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));
    if (size != NRF)
    {
        std::printf(
            "error: expected array<%zd>, received array<%zd>", NRF, size);
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

std::int64_t load_i(std::istream& is, std::string const var_name)
{
    std::int64_t i{};
    is.read(reinterpret_cast<char*>(&i), sizeof(std::int64_t));

    std::printf("loaded int64_t %s.\n", var_name.c_str());

    return i;
}

double load_d(std::istream& is, std::string const var_name)
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
        std::printf("error: cannot open args file \"%s\".", args_fn.c_str());
        std::abort();
    }

    fx_args args;
    args.opts_eos = load_i(is, "opts_eos");
    args.opts_problem = load_i(is, "opts_problem");
    args.opts_dual_energy_sw1 = load_d(is, "opts_dual_energy_sw1");
    args.opts_dual_energy_sw2 = load_d(is, "opts_dual_energy_sw2");
    args.physcon_A = load_d(is, "physcon_A");
    args.physcon_B = load_d(is, "physcon_B");
    args.physcon_c = load_d(is, "physcon_c");
    args.er_i = load_i(is, "er_i");
    args.fx_i = load_i(is, "fx_i");
    args.fy_i = load_i(is, "fy_i");
    args.fz_i = load_i(is, "fz_i");
    args.d = load_i(is, "d");
    args.rho = load_v(is, "rho");
    args.sx = load_v(is, "sx");
    args.sy = load_v(is, "sy");
    args.sz = load_v(is, "sz");
    args.egas = load_v(is, "egas");
    args.tau = load_v(is, "tau");
    args.fgamma = load_d(is, "fgamma");
    args.U = load_a(is, "U");
    args.mmw = load_v(is, "mmw");
    args.X_spc = load_v(is, "X_spc");
    args.Z_spc = load_v(is, "Z_spc");
    args.dt = load_d(is, "dt");
    args.clightinv = load_d(is, "clightinv");

    if (is.eof())
    {
        std::printf("error: end of file not reached. tell: %zd\n",
            static_cast<std::size_t>(is.tellg()));
        std::abort();
    }
    std::printf("read eof of \"%s\". tell: %zd\n",
        args_fn.c_str(),
        static_cast<std::size_t>(is.tellg()));

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
        std::printf("error: cannot open outs file \"%s\".", outs_fn.c_str());
        std::abort();
    }

    fx_outs outs;
    outs.sx = std::move(load_v(is, "sx"));
    outs.sy = std::move(load_v(is, "sy"));
    outs.sz = std::move(load_v(is, "sz"));
    outs.egas = std::move(load_v(is, "egas"));
    outs.U = std::move(load_a(is, "U"));

    if (is.eof())
    {
        std::printf("error: end of file not reached. tell: %zd\n",
            static_cast<std::size_t>(is.tellg()));
        std::abort();
    }
    std::printf("reached eof of \"%s\". tell: %zd\n",
        outs_fn.c_str(),
        static_cast<std::size_t>(is.tellg()));

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
