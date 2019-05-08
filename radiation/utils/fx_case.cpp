#include "config.hpp"
#include "utils/fx_case.hpp"
#include "utils/util.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <ios>
#include <string>
#include <type_traits>
#include <vector>

constexpr char const* const basepath = OCTORAD_DUMP_DIR "/octotiger-radiation-";

namespace octotiger {
    struct throw_on_pos_mismatch
    {
        void operator()(std::istream& is)
        {
            std::streampos orig_pos{};
            std::streampos actual_pos = is.tellg();
            is.read(reinterpret_cast<char*>(&orig_pos), sizeof(std::streampos));

            if (actual_pos != orig_pos)
            {
                throw formatted_exception(
                    "error: stream positions do not match. actual: %, original: %",
                    actual_pos,
                    orig_pos);
            }
        }
    };

    struct ignore_streampos
    {
        void operator()(std::istream& is)
        {
            is.ignore(24);
        }
    };

    void handle_streampos(std::istream& is)
    {
        // HACK: I did not consider streampos having different sizes on win, linux,
        // and mac when creating dump files
        using streampos_handler_t =
            std::conditional<sizeof(std::streampos) == 24,
                throw_on_pos_mismatch,
                ignore_streampos>::type;
        streampos_handler_t{}(is);
    }

    std::vector<double> load_v(std::istream& is)
    {
        handle_streampos(is);

        std::size_t size{};
        is.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));

        std::vector<double> v(size);
        is.read(reinterpret_cast<char*>(&v[0]), size * sizeof(double));

        return v;
    }

    std::array<std::vector<double>, NRF> load_a(std::istream& is)
    {
        handle_streampos(is);

        std::size_t size{};
        is.read(reinterpret_cast<char*>(&size), sizeof(std::size_t));
        if (size != NRF)
        {
            throw formatted_exception(
                "error: expected array<%>, received array<%>", NRF, size);
        }

        std::array<std::vector<double>, NRF> a{};
        for (auto& e : a)
        {
            e = load_v(is);
        }

        return a;
    }

    std::int64_t load_i(std::istream& is)
    {
        handle_streampos(is);

        std::int64_t i{};
        is.read(reinterpret_cast<char*>(&i), sizeof(std::int64_t));

        return i;
    }

    double load_d(std::istream& is)
    {
        handle_streampos(is);

        double d{};
        is.read(reinterpret_cast<char*>(&d), sizeof(double));

        return d;
    }

    fx_args load_case_args(std::size_t index)
    {
        std::string const args_fn = std::string{basepath} +
            std::to_string(index) + std::string{".args"};
        std::ifstream is{args_fn, std::ios::binary};

        if (!is)
        {
            throw formatted_exception(
                "error: cannot open args file \"%\".", args_fn.c_str());
        }

        fx_args args;
        args.opts_eos = load_i(is);
        args.opts_problem = load_i(is);
        args.opts_dual_energy_sw1 = load_d(is);
        args.opts_dual_energy_sw2 = load_d(is);
        args.physcon_A = load_d(is);
        args.physcon_B = load_d(is);
        args.physcon_c = load_d(is);
        args.er_i = load_i(is);
        args.fx_i = load_i(is);
        args.fy_i = load_i(is);
        args.fz_i = load_i(is);
        args.d = load_i(is);
        args.rho = load_v(is);
        args.sx = load_v(is);
        args.sy = load_v(is);
        args.sz = load_v(is);
        args.egas = load_v(is);
        args.tau = load_v(is);
        args.fgamma = load_d(is);
        args.U = load_a(is);
        args.mmw = load_v(is);
        args.X_spc = load_v(is);
        args.Z_spc = load_v(is);
        args.dt = load_d(is);
        args.clightinv = load_d(is);

        if (is.eof())
        {
            throw formatted_exception(
                "error: end of file not reached. tell: %\n", is.tellg());
        }

        return args;
    }

    fx_outs load_case_outs(std::size_t index)
    {
        std::string const outs_fn = std::string{basepath} +
            std::to_string(index) + std::string{".outs"};
        std::ifstream is{outs_fn, std::ios::binary};

        if (!is)
        {
            throw formatted_exception(
                "error: cannot open outs file \"%\".", outs_fn.c_str());
        }

        fx_outs outs;
        outs.sx = load_v(is);
        outs.sy = load_v(is);
        outs.sz = load_v(is);
        outs.egas = load_v(is);
        //outs.tau = load_v(is);
        outs.U = load_a(is);

        if (is.eof())
        {
            throw formatted_exception(
                "error: end of file not reached. tell: %\n", is.tellg());
        }

        return outs;
    }

    fx_case import_case(std::size_t index)
    {
        fx_case ret;

        ret.index = index;
        ret.args = load_case_args(index);
        ret.outs = load_case_outs(index);

        return ret;
    }
}
