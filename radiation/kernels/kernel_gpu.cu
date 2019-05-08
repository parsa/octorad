#include "config.hpp"
#include "kernels/kernel_gpu.hpp"
#include "kernels/helpers.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <sstream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

constexpr std::size_t RAD_GRID_I = RAD_NX - (2 * RAD_BW);

struct space_vector
{
    explicit space_vector() = default;

    __device__ space_vector(space_vector const& other)
    {
        for (std::size_t i = 0; i < 4; ++i)
        {
            data_[i] = other.data_[i];
        }
    }

    __device__ double& operator[](std::size_t const index)
    {
        return data_[index];
    }

    __device__ double operator[](std::size_t const index) const
    {
        return data_[index];
    }

    __device__ space_vector operator/(double const rhs) const
    {
        space_vector ret;
        ret.data_[0] = data_[0] / rhs;
        ret.data_[1] = data_[1] / rhs;
        ret.data_[2] = data_[2] / rhs;
        ret.data_[3] = data_[3] / rhs;

        return ret;
    }

    __device__ space_vector operator-(space_vector const& rhs) const
    {
        space_vector ret;
        ret.data_[0] = data_[0] - rhs.data_[0];
        ret.data_[1] = data_[1] - rhs.data_[1];
        ret.data_[2] = data_[2] - rhs.data_[2];
        ret.data_[3] = data_[3] - rhs.data_[3];

        return ret;
    }

    __device__ space_vector operator*(double const rhs) const
    {
        space_vector ret;
        ret.data_[0] = data_[0] * rhs;
        ret.data_[1] = data_[1] * rhs;
        ret.data_[2] = data_[2] * rhs;
        ret.data_[3] = data_[3] * rhs;

        return ret;
    }

private:
    double data_[4]{};
};

__device__ space_vector make_space_vector(double x, double y, double z)
{
    space_vector ret;
    ret[0] = x;
    ret[1] = y;
    ret[2] = z;
    return ret;
}

template <typename T1, typename T2>
struct d_pair
{
    explicit d_pair() = default;
    __device__ d_pair(T1 f, T2 s) : first(f), second(s)
    {
    }

    T1 first{};
    T2 second{};
};

__device__ __host__ inline std::int64_t hindex(
    std::int64_t i, std::int64_t j, std::int64_t k)
{
    return i * H_DNX + j * H_DNY + k * H_DNZ;
}

__device__ inline double INVERSE(double a)
{
    return 1.0 / a;
}

template <typename T>
__device__ T B_p(
    std::int64_t const opts_problem, double const physcon_c, T rho, T e, T mmw)
{
    assert(opts_problem == MARSHAK);
    return T((physcon_c / 4.0 / M_PI)) * e;
}

template <typename T>
__device__ T kappa_p(
    std::int64_t const opts_problem, T rho, T e, T mmw, double X, double Z)
{
    assert(opts_problem == MARSHAK);
    return MARSHAK_OPAC;
}

template <typename T>
__device__ T kappa_R(
    std::int64_t const opts_problem, T rho, T e, T mmw, double X, double Z)
{
    assert(opts_problem == MARSHAK);
    return MARSHAK_OPAC;
}

template <typename T>
__device__ inline T sqr(T const& val)
{
    return val * val;
}

template <typename T>
__device__ __host__ inline T cube(T const& val)
{
    return val * val * val;
}

__device__ inline double ztwd_enthalpy(
    double const physcon_A, double const physcon_B, double d)
{
    double A = physcon_A;
    double B = physcon_B;
    if (d < 0.0)
    {
        std::printf("d = %e in ztwd_enthalpy\n", d);
        assert(false);
    }
    double const x = pow(d / B, 1.0 / 3.0);
    double h;
    if (x < 0.01)
    {
        h = 4.0 * A / B * sqr(x);
    }
    else
    {
        h = 8.0 * A / B * (sqrt(sqr(x) + 1.0) - 1.0);
    }
    return h;
}

__device__ inline double ztwd_pressure(
    double const physcon_A, double const physcon_B, double d)
{
    double A = physcon_A;
    double B = physcon_B;
    double const x = pow(d / B, 1.0 / 3.0);
    double p;
    if (x < 0.01)
    {
        p = 1.6 * A * sqrt(x) * cube(x);
    }
    else
    {
        p = A *
            (x * (2.0 * sqrt(x) - 3.0) * sqrt(sqrt(x) + 1.0) + 3.0 * asinh(x));
    }
    return p;
}

__device__ double ztwd_energy(
    double const physcon_A, double const physcon_B, double d)
{
    return fmax(ztwd_enthalpy(physcon_A, physcon_B, d) * d -
            ztwd_pressure(physcon_A, physcon_B, d),
        double(0));
}

__device__ inline double& e_ij(
    double* U, std::size_t Ui_size, std::size_t i, std::size_t j)
{
    return U[i * Ui_size + j];
}

template <typename F>
__device__ void abort_if_solver_not_converged(double const eg_t0, double E0, F const test,
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
            fmax(fabs(f_mid), fabs(f_min)) / (E + eg_t);
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
        //std::printf("iteration: %d, error: %g\n", i, error);
    }
    // Error is not smaller that error tolerance after performed iterations. Abort.
    std::printf("implicit radiation solver failed to converge\n");
    // Fail if code reaches here
    assert(false);
}

__device__ d_pair<double, space_vector> implicit_radiation_step(
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
    double ei{};
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
        ei = fmax(eg_t0 - E + E0 - 0.5 * u2, double{});
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

    return d_pair<double, space_vector>(
        double((E - E0) * dtinv * rhoc2), ((F - F0) * dtinv * rhoc2 * c));
}

__global__ void __launch_bounds__(512, 1) radiation_impl(
    std::int64_t const opts_eos,
    std::int64_t const opts_problem,
    double const opts_dual_energy_sw1,
    double const opts_dual_energy_sw2,
    double const physcon_A,
    double const physcon_B,
    double const physcon_c,
    std::int64_t const er_i,
    std::int64_t const fx_i,
    std::int64_t const fy_i,
    std::int64_t const fz_i,
    std::int64_t const d,
    std::size_t const grid_i_size,
    std::size_t const Ui_size,
    double const* const rho,
    double* const sx,
    double* const sy,
    double* const sz,
    double* const egas,
    double* const tau,
    double const fgamma,
    double* const U,
    double const* const mmw,
    double const* const X_spc,
    double const* const Z_spc,
    double const dt,
    double const clightinv)
{
    std::int64_t const i = threadIdx.x;
    std::int64_t const j = threadIdx.y;
    std::int64_t const k = threadIdx.z;

    std::int64_t const iiih = (i * grid_i_size + j) * grid_i_size + k;
    std::int64_t const iiir = (i * grid_i_size + j) * grid_i_size + k;
    double const den = rho[iiih];
    double const deninv = INVERSE(den);
    double vx = sx[iiih] * deninv;
    double vy = sy[iiih] * deninv;
    double vz = sz[iiih] * deninv;

    // Compute e0 from dual energy formalism
    double e0 = egas[iiih]
       - 0.5 * vx * vx * den
       - 0.5 * vy * vy * den
       - 0.5 * vz * vz * den;
    if (opts_eos == eos_wd)
    {
        e0 -= ztwd_energy(physcon_A, physcon_B, den);
    }
    if (e0 < egas[iiih] * opts_dual_energy_sw2)
    {
        e0 = pow(tau[iiih], fgamma);
    }
    double E0 = e_ij(U, Ui_size, er_i, iiir);
    space_vector F0 = make_space_vector(
        e_ij(U, Ui_size, fx_i, iiir),
        e_ij(U, Ui_size, fy_i, iiir),
        e_ij(U, Ui_size, fz_i, iiir));
    space_vector u0 = make_space_vector(vx, vy, vz);
    double E1 = E0;
    space_vector F1 = F0;
    space_vector u1 = u0;
    double e1 = e0;

    auto const ddt = implicit_radiation_step(opts_problem, physcon_c, E1, e1,
        F1, u1, den, mmw[iiir], X_spc[iiir], Z_spc[iiir], dt);
    double const dE_dt = ddt.first;
    double const dFx_dt = ddt.second[0];
    double const dFy_dt = ddt.second[1];
    double const dFz_dt = ddt.second[2];

    // Accumulate derivatives
    e_ij(U, Ui_size, er_i, iiir) += dE_dt * dt;
    e_ij(U, Ui_size, fx_i, iiir) += dFx_dt * dt;
    e_ij(U, Ui_size, fy_i, iiir) += dFy_dt * dt;
    e_ij(U, Ui_size, fz_i, iiir) += dFz_dt * dt;

    egas[iiih] -= dE_dt * dt;
    sx[iiih] -= dFx_dt * dt * clightinv * clightinv;
    sy[iiih] -= dFy_dt * dt * clightinv * clightinv;
    sz[iiih] -= dFz_dt * dt * clightinv * clightinv;

    // Find tau with dual energy formalism
    double e = egas[iiih]
      - 0.5 * sx[iiih] * sx[iiih] * deninv
      - 0.5 * sy[iiih] * sy[iiih] * deninv
      - 0.5 * sz[iiih] * sz[iiih] * deninv;
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
    if (e_ij(U, Ui_size, er_i, iiir) <= 0.0)
    {
        std::printf("Er = %e %e %e %e\n", E0, E1, e_ij(U, Ui_size, er_i, iiir), dt);
        assert(false);
    }
    e = fmax(e, 0.0);
    tau[iiih] = std::pow(e, INVERSE(fgamma));
    if (e_ij(U, Ui_size, er_i, iiir) <= 0.0)
    {
        std::printf("2231242!!! %e %e %e \n", E0, e_ij(U, Ui_size, er_i, iiir), dE_dt * dt);
        assert(false);
    }
    if (opts_problem == MARSHAK)
    {
        sx[iiih] = sy[iiih] = sz[iiih] = 0.0;
        egas[iiih] = e;
    }
}

namespace octotiger {
    void radiation_gpu_kernel(std::int64_t const opts_eos,
        std::int64_t const opts_problem,
        double const opts_dual_energy_sw1,
        double const opts_dual_energy_sw2,
        double const physcon_A,
        double const physcon_B,
        double const physcon_c,
        std::int64_t const er_i,
        std::int64_t const fx_i,
        std::int64_t const fy_i,
        std::int64_t const fz_i,
        std::int64_t const d,
        std::vector<double> const& rho,
        std::vector<double>& sx,
        std::vector<double>& sy,
        std::vector<double>& sz,
        std::vector<double>& egas,
        std::vector<double>& tau,
        double const fgamma,
        std::array<std::vector<double>, NRF>
            U,
        std::vector<double> const mmw,
        std::vector<double> const X_spc,
        std::vector<double> const Z_spc,
        double const dt,
        double const clightinv)
    {
        std::size_t const grid_array_size = cube(RAD_GRID_I);
        std::vector<double> h_rho(grid_array_size);
        std::vector<double> h_sx(grid_array_size);
        std::vector<double> h_sy(grid_array_size);
        std::vector<double> h_sz(grid_array_size);
        std::vector<double> h_egas(grid_array_size);
        std::vector<double> h_tau(grid_array_size);
        std::vector<double> h_U(NRF * grid_array_size);
        std::vector<double> h_mmw(grid_array_size);
        std::vector<double> h_X_spc(grid_array_size);
        std::vector<double> h_Z_spc(grid_array_size);

        {
            std::size_t index_counter{};
            for (std::size_t i = RAD_BW; i != RAD_NX - RAD_BW; ++i)
            {
                for (std::size_t j = RAD_BW; j != RAD_NX - RAD_BW; ++j)
                {
                    for (std::size_t k = RAD_BW; k != RAD_NX - RAD_BW; ++k)
                    {
                        std::size_t const iiih = hindex(i + d, j + d, k + d);
                        h_rho[index_counter] = rho[iiih];
                        h_sx[index_counter] = sx[iiih];
                        h_sy[index_counter] = sy[iiih];
                        h_sz[index_counter] = sz[iiih];
                        h_egas[index_counter] = egas[iiih];
                        h_tau[index_counter] = tau[iiih];
                        for (std::size_t l = 0; l < NRF; ++l)
                        {
                            h_U[l * grid_array_size + index_counter] = U[l][iiih];
                        }
                        h_mmw[index_counter] = mmw[iiih];
                        h_X_spc[index_counter] = X_spc[iiih];
                        h_Z_spc[index_counter] = Z_spc[iiih];

                        ++index_counter;
                    }
                }
            }
        }

        double* d_rho = alloc_copy_device(h_rho);
        double* d_sx = alloc_copy_device(h_sx);
        double* d_sy = alloc_copy_device(h_sy);
        double* d_sz = alloc_copy_device(h_sz);
        double* d_egas = alloc_copy_device(h_egas);
        double* d_tau = alloc_copy_device(h_tau);
        double* d_U = alloc_copy_device(h_U);
        double* d_mmw = alloc_copy_device(h_mmw);
        double* d_X_spc = alloc_copy_device(h_X_spc);
        double* d_Z_spc = alloc_copy_device(h_Z_spc);

        //cudaLaunchKernel(radiation_impl, 1, 1, args, 0, 0)
        // NOTE: too many registers (currently 168)
        radiation_impl<<<1, dim3(RAD_GRID_I, RAD_GRID_I, RAD_GRID_I)>>>(
            opts_eos,
            opts_problem,
            opts_dual_energy_sw1,
            opts_dual_energy_sw2,
            physcon_A,
            physcon_B,
            physcon_c,
            er_i,
            fx_i,
            fy_i,
            fz_i,
            d,
            RAD_GRID_I,
            grid_array_size,
            d_rho,
            d_sx,
            d_sy,
            d_sz,
            d_egas,
            d_tau,
            fgamma,
            d_U,
            d_mmw,
            d_X_spc,
            d_Z_spc,
            dt,
            clightinv
            );
        throw_if_cuda_error();

        copy_from_device(d_sx, h_sx);
        copy_from_device(d_sy, h_sy);
        copy_from_device(d_sz, h_sz);
        copy_from_device(d_egas, h_egas);
        copy_from_device(d_U, h_U);

        {
            std::size_t index_counter{};
            for (std::size_t i = RAD_BW; i < RAD_NX - RAD_BW; ++i)
            {
                for (std::size_t j = RAD_BW; j < RAD_NX - RAD_BW; ++j)
                {
                    for (std::size_t k = RAD_BW; k < RAD_NX - RAD_BW; ++k)
                    {
                        std::size_t const iiih = hindex(i + d, j + d, k + d);
                        sx[iiih] = h_sx[index_counter];
                        sy[iiih] = h_sy[index_counter];
                        sz[iiih] = h_sz[index_counter];
                        egas[iiih] = h_egas[index_counter];
                        for (std::size_t l = 0; l < NRF; ++l)
                        {
                            U[l][iiih] = h_U[l * grid_array_size + index_counter];
                        }

                        ++index_counter;
                    }
                }
            }
        }

        free_device(d_rho);
        free_device(d_sx);
        free_device(d_sy);
        free_device(d_sz);
        free_device(d_egas);
        free_device(d_tau);
        free_device(d_U);
        free_device(d_mmw);
        free_device(d_X_spc);
        free_device(d_Z_spc);
    }
}
