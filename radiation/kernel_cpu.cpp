#include "config.hpp"
#include "kernel_cpu.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#if defined(OCTORAD_HAVE_VC)
    #include <Vc/Vc>
#endif

#if defined(OCTORAD_HAVE_VC)
using space_vector = Vc::Vector<double, Vc::VectorAbi::Avx>;
#else
using space_vector = std::array<double, 4>;

space_vector operator/(space_vector const& lhs, double const rhs)
{
    space_vector ret;
    ret[0] = lhs[0] / rhs;
    ret[1] = lhs[1] / rhs;
    ret[2] = lhs[2] / rhs;
    ret[3] = lhs[3] / rhs;

    return ret;
}

space_vector operator-(space_vector const& lhs, space_vector const& rhs)
{
    space_vector ret;
    ret[0] = lhs[0] - rhs[0];
    ret[1] = lhs[1] - rhs[1];
    ret[2] = lhs[2] - rhs[2];
    ret[3] = lhs[3] - rhs[3];

    return ret;
}

space_vector operator*(space_vector const& lhs, double const rhs)
{
    space_vector ret;
    ret[0] = lhs[0] * rhs;
    ret[1] = lhs[1] * rhs;
    ret[2] = lhs[2] * rhs;
    ret[3] = lhs[3] * rhs;

    return ret;
}
#endif

constexpr std::int64_t MARSHAK = 9;

constexpr inline std::int64_t hindex(
    std::int64_t i, std::int64_t j, std::int64_t k)
{
    return i * H_DNX + j * H_DNY + k * H_DNZ;
}

constexpr inline std::int64_t rindex(
    std::int64_t x, std::int64_t y, std::int64_t z)
{
    return z + RAD_NX * (y + RAD_NX * x);
}

constexpr inline double INVERSE(double a)
{
    return 1.0 / a;
}

template <class U>
U B_p(
    std::int64_t const opts_problem, double const physcon_c, U rho, U e, U mmw)
{
    if (opts_problem == MARSHAK)
    {
        return U((physcon_c / 4.0 / M_PI)) * e;
    }
    std::printf("error: marshak is the only supported problem type\n");
    std::abort();
}

template <class U>
U kappa_p(
    std::int64_t const opts_problem, U rho, U e, U mmw, double X, double Z)
{
    if (opts_problem == MARSHAK)
    {
        return MARSHAK_OPAC;
    }
    std::printf("error: marshak is the only supported problem type\n");
    std::abort();
}

template <class U>
U kappa_R(
    std::int64_t const opts_problem, U rho, U e, U mmw, double X, double Z)
{
    if (opts_problem == MARSHAK)
    {
        return MARSHAK_OPAC;
    }
    std::printf("error: marshak is the only supported problem type\n");
    std::abort();
}

template <typename T>
constexpr inline T sqr(T const& val)
{
    return val * val;
}

template <typename T>
constexpr inline T cube(T const& val)
{
    return val * val * val;
}

inline double ztwd_enthalpy(
    double const physcon_A, double const physcon_B, double d)
{
    double A = physcon_A;
    double B = physcon_B;
    if (d < 0.0)
    {
        std::printf("d = %e in ztwd_enthalpy\n", d);
        std::abort();
    }
    double const x = std::pow(d / B, 1.0 / 3.0);
    double h;
    if (x < 0.01)
    {
        h = 4.0 * A / B * sqr(x);
    }
    else
    {
        h = 8.0 * A / B * (std::sqrt(sqr(x) + 1.0) - 1.0);
    }
    return h;
}

inline double ztwd_pressure(
    double const physcon_A, double const physcon_B, double d)
{
    double A = physcon_A;
    double B = physcon_B;
    double const x = std::pow(d / B, 1.0 / 3.0);
    double p;
    if (x < 0.01)
    {
        p = 1.6 * A * std::sqrt(x) * cube(x);
    }
    else
    {
        p = A *
            (x * (2.0 * std::sqrt(x) - 3.0) * std::sqrt(std::sqrt(x) + 1.0) +
                3.0 * std::asinh(x));
    }
    return p;
}

double ztwd_energy(double const physcon_A, double const physcon_B, double d)
{
    double A = physcon_A;
    double B = physcon_B;
    return std::max(ztwd_enthalpy(physcon_A, physcon_B, d) * d -
            ztwd_pressure(physcon_A, physcon_B, d),
        double(0));
}

template <typename F>
void abort_if_solver_not_converged(double const eg_t0, double E0, F const test,
    double const E, double const eg_t)
{
    // Bisection root finding method
    // Indices of max, mid, and min
    double de_max = eg_t0;
    double de_mid = 0.0;
    double de_min = -E0;
    // Values of max, mid, and min
    double f_min = test(de_min);
    double f_mid = test(de_mid);
    // Max iterations
    constexpr std::size_t max_iterations = 50;
    // Errors
    double const error_tolerance = 1.0e-9;

    for (std::size_t i = 0; i < max_iterations; ++i)
    {
        // Root solver has converged if error is smaller that error tolerance
        double const error =
            std::max(std::abs(f_mid), std::abs(f_min)) / (E + eg_t);
        if (error < error_tolerance)
        {
            return;
        }

        // If signs don't match, continue search in the lower half
        if ((f_min < 0) != (f_mid < 0))
        {
            de_max = de_mid;
            de_mid = 0.5 * (de_min + de_max);
            f_mid = test(de_mid);
        }
        // Continue search in the upper half
        else
        {
            de_min = de_mid;
            de_mid = 0.5 * (de_min + de_max);
            f_min = f_mid;
            f_mid = test(de_mid);
        }
    }
    // Error is not smaller that error tolerance after performed iterations. Abort.
    std::printf("Implicit radiation solver failed to converge\n");
    std::abort();
}

std::pair<double, space_vector> implicit_radiation_step(
    std::int64_t const opts_problem, double const physcon_c, double E0,
    double& e0, space_vector F0, space_vector u0, double const rho,
    double const mmw, double const X, double const Z, double const dt)
{
    double const c = physcon_c;
    double kp = kappa_p(opts_problem, rho, e0, mmw, X, Z);
    double kr = kappa_R(opts_problem, rho, e0, mmw, X, Z);
    double const rhoc2 = rho * c * c;

    E0 /= rhoc2;
    F0 = F0 / (rhoc2 * c);
    e0 /= rhoc2;
    u0 = u0 / c;
    kp *= dt * c;
    kr *= dt * c;

    auto const B = [opts_problem, physcon_c, rho, mmw, c, rhoc2](
                       double const e) {
        return (4.0 * M_PI / c) *
            B_p(opts_problem, physcon_c, rho, e * rhoc2, mmw) / rhoc2;
    };

    auto E = E0;
    auto eg_t = e0 + 0.5 * (u0[0] * u0[0] + u0[1] * u0[1] + u0[2] * u0[2]);
    auto F = F0;
    auto u = u0;
    double ei;
    auto const eg_t0 = eg_t;

    double u2_0 = 0.0;
    double F2_0 = 0.0;
    for (int d = 0; d < NDIM; d++)
    {
        u2_0 += u[d] * u[d];
        F2_0 += F[d] * F[d];
    }
    // printf( "%e %e\n", (double) u2_0, (double) (F2_0/E/E));
    auto const test = [&](double de) {
        E = E0 + de;
        double u2 = 0.0;
        double udotF = 0.0;
        for (int d = 0; d < NDIM; d++)
        {
            auto const num = F0[d] + (4.0 / 3.0) * kr * E * (u0[d] + F0[d]);
            auto const den = 1.0 + kr * (1.0 + (4.0 / 3.0) * E);
            auto const deninv = 1.0 / den;
            F[d] = num * deninv;
            u[d] = u0[d] + F0[d] - F[d];
            u2 += u[d] * u[d];
            udotF += F[d] * u[d];
        }
        ei = std::max(eg_t0 - E + E0 - 0.5 * u2, double{});
        double const b = B(ei);
        double f = E - E0;
        f += (kp * (E - b) + (kr - 2.0 * kp) * udotF);
        eg_t = eg_t0 + E0 - E;
        return f;
    };

    abort_if_solver_not_converged(eg_t0, E0, test, E, eg_t);

    ei = eg_t - 0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    e0 = ei * rhoc2;
    auto const dtinv = 1.0 / dt;

    return std::make_pair(
        double((E - E0) * dtinv * rhoc2), ((F - F0) * dtinv * rhoc2 * c));
}

void radiation_cpu_kernel(std::int64_t const opts_eos,
    std::int64_t const opts_problem, double const opts_dual_energy_sw1,
    double const opts_dual_energy_sw2, double const physcon_A,
    double const physcon_B, double const physcon_c, std::int64_t const er_i,
    std::int64_t const fx_i, std::int64_t const fy_i, std::int64_t const fz_i,
    std::int64_t const d, std::vector<double> const& rho,
    std::vector<double>& sx, std::vector<double>& sy, std::vector<double>& sz,
    std::vector<double>& egas, std::vector<double>& tau, double const fgamma,
    std::array<std::vector<double>, NRF> U, std::vector<double> const mmw,
    std::vector<double> const X_spc, std::vector<double> const Z_spc,
    double const dt, double const clightinv)
{
    for (std::int64_t i = RAD_BW; i != RAD_NX - RAD_BW; ++i)
    {
        for (std::int64_t j = RAD_BW; j != RAD_NX - RAD_BW; ++j)
        {
            for (std::int64_t k = RAD_BW; k != RAD_NX - RAD_BW; ++k)
            {
                std::int64_t const iiih = hindex(i + d, j + d, k + d);
                std::int64_t const iiir = rindex(i, j, k);
                double const den = rho[iiih];
                double const deninv = INVERSE(den);
                double vx = sx[iiih] * deninv;
                double vy = sy[iiih] * deninv;
                double vz = sz[iiih] * deninv;

                // Compute e0 from dual energy formalism
                double e0 = egas[iiih];
                e0 -= 0.5 * vx * vx * den;
                e0 -= 0.5 * vy * vy * den;
                e0 -= 0.5 * vz * vz * den;
                if (opts_eos == eos_wd)
                {
                    e0 -= ztwd_energy(physcon_A, physcon_B, den);
                }
                if (e0 < egas[iiih] * opts_dual_energy_sw2)
                {
                    e0 = std::pow(tau[iiih], fgamma);
                }
                double E0 = U[er_i][iiir];
                space_vector F0;
                space_vector u0;
                F0[0] = U[fx_i][iiir];
                F0[1] = U[fy_i][iiir];
                F0[2] = U[fz_i][iiir];
                u0[0] = vx;
                u0[1] = vy;
                u0[2] = vz;
                double E1 = E0;
                space_vector F1 = F0;
                space_vector u1 = u0;
                double e1 = e0;

                auto const ddt =
                    implicit_radiation_step(opts_problem, physcon_c, E1, e1, F1,
                        u1, den, mmw[iiir], X_spc[iiir], Z_spc[iiir], dt);
                double const dE_dt = ddt.first;
                double const dFx_dt = ddt.second[0];
                double const dFy_dt = ddt.second[1];
                double const dFz_dt = ddt.second[2];

                // Accumulate derivatives
                U[er_i][iiir] += dE_dt * dt;
                U[fx_i][iiir] += dFx_dt * dt;
                U[fy_i][iiir] += dFy_dt * dt;
                U[fz_i][iiir] += dFz_dt * dt;

                egas[iiih] -= dE_dt * dt;
                sx[iiih] -= dFx_dt * dt * clightinv * clightinv;
                sy[iiih] -= dFy_dt * dt * clightinv * clightinv;
                sz[iiih] -= dFz_dt * dt * clightinv * clightinv;

                // Find tau with dual energy formalism
                double e = egas[iiih];
                e -= 0.5 * sx[iiih] * sx[iiih] * deninv;
                e -= 0.5 * sy[iiih] * sy[iiih] * deninv;
                e -= 0.5 * sz[iiih] * sz[iiih] * deninv;
                if (opts_eos == eos_wd)
                {
                    e -= ztwd_energy(physcon_A, physcon_B, den);
                }
                if (e < opts_dual_energy_sw1 * egas[iiih])
                {
                    e = e1;
                }
                if (opts_problem == MARSHAK)
                {
                    egas[iiih] = e;
                    sx[iiih] = sy[iiih] = sz[iiih] = 0;
                }
                if (U[er_i][iiir] <= 0.0)
                {
                    std::printf(
                        "Er = %e %e %e %e\n", E0, E1, U[er_i][iiir], dt);
                    std::abort();
                }
                e = std::max(e, 0.0);
                tau[iiih] = std::pow(e, INVERSE(fgamma));
                if (U[er_i][iiir] <= 0.0)
                {
                    std::printf("2231242!!! %e %e %e \n", E0, U[er_i][iiir],
                        dE_dt * dt);
                    std::abort();
                }
                if (opts_problem == MARSHAK)
                {
                    sx[iiih] = sy[iiih] = sz[iiih] = 0;
                    egas[iiih] = e;
                }
            }
        }
    }
}
