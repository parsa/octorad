#include "config.hpp"
#include "kernel_gpu.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

constexpr std::size_t loop_iterations = RAD_NX - (2 * RAD_BW);

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
        ret[0] = data_[0] / rhs;
        ret[1] = data_[1] / rhs;
        ret[2] = data_[2] / rhs;
        ret[3] = data_[3] / rhs;

        return ret;
    }

    __device__ space_vector operator-(space_vector const& rhs) const
    {
        space_vector ret;
        ret[0] = data_[0] - rhs.data_[0];
        ret[1] = data_[1] - rhs.data_[1];
        ret[2] = data_[2] - rhs.data_[2];
        ret[3] = data_[3] - rhs.data_[3];

        return ret;
    }

    __device__ space_vector operator*(double const rhs) const
    {
        space_vector ret;
        ret[0] = data_[0] * rhs;
        ret[1] = data_[1] * rhs;
        ret[2] = data_[2] * rhs;
        ret[3] = data_[3] * rhs;

        return ret;
    }

private:
    double data_[4]{};
};

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

__device__ inline std::int64_t hindex(
    std::int64_t i, std::int64_t j, std::int64_t k)
{
    return i * H_DNX + j * H_DNY + k * H_DNZ;
}

__device__ inline std::int64_t rindex(
    std::int64_t x, std::int64_t y, std::int64_t z)
{
    return z + RAD_NX * (y + RAD_NX * x);
}

__device__ inline double INVERSE(double a)
{
    return 1.0 / a;
}

template <class U>
__device__ U B_p(
    std::int64_t const opts_problem, double const physcon_c, U rho, U e, U mmw)
{
    assert(opts_problem == MARSHAK);
    return U((physcon_c / 4.0 / M_PI)) * e;
}

template <class U>
__device__ U kappa_p(
    std::int64_t const opts_problem, U rho, U e, U mmw, double X, double Z)
{
    assert(opts_problem == MARSHAK);
    return MARSHAK_OPAC;
}

template <class U>
__device__ U kappa_R(
    std::int64_t const opts_problem, U rho, U e, U mmw, double X, double Z)
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
__device__ inline T cube(T const& val)
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
    double A = physcon_A;
    double B = physcon_B;
    return fmax(ztwd_enthalpy(physcon_A, physcon_B, d) * d -
            ztwd_pressure(physcon_A, physcon_B, d),
        double(0));
}

__device__ inline double& Uij(
    double* U, std::size_t Ui_size, std::size_t i, std::size_t j)
{
    return U[i * Ui_size + j];
}

void abort_if_cuda_error(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::printf("cuda error: %s\n", cudaGetErrorString(err));
        assert(false);
    }
}

void abort_if_cuda_error()
{
    cudaError_t err = cudaGetLastError();
    abort_if_cuda_error(err);
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
    }
    // Error is not smaller that error tolerance after performed iterations. Abort.
    std::printf("Implicit radiation solver failed to converge\n");
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

__global__ void radiation_impl(
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
    double const* const rho, std::size_t rho_size,
    double* const sx, std::size_t sx_size,
    double* const sy, std::size_t sy_size,
    double* const sz, std::size_t sz_size,
    double* const egas, std::size_t egas_size,
    double* const tau, std::size_t tau_size,
    double const fgamma,
    double* const U, std::size_t Ui_size,
    double const* const mmw, std::size_t mmw_size,
    double const* const X_spc, std::size_t X_spc_size,
    double const* const Z_spc, std::size_t Z_spc_size,
    double const dt,
    double const clightinv)
{
    std::int64_t i = threadIdx.x;
    std::int64_t j = threadIdx.y;
    std::int64_t k = threadIdx.z;

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
    //    e0 -= ztwd_energy(physcon_A, physcon_B, den);
    }
    if (e0 < egas[iiih] * opts_dual_energy_sw2)
    {
    //    e0 = std::pow(tau[iiih], fgamma);
    }
    double E0 = Uij(U, Ui_size, er_i, iiir);
    space_vector F0;
    space_vector u0;
    F0[0] = Uij(U, Ui_size, fx_i, iiir);
    F0[1] = Uij(U, Ui_size, fy_i, iiir);
    F0[2] = Uij(U, Ui_size, fz_i, iiir);
    u0[0] = vx;
    u0[1] = vy;
    u0[2] = vz;
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
    Uij(U, Ui_size, er_i, iiir) += dE_dt * dt;
    Uij(U, Ui_size, fx_i, iiir) += dFx_dt * dt;
    Uij(U, Ui_size, fy_i, iiir) += dFy_dt * dt;
    Uij(U, Ui_size, fz_i, iiir) += dFz_dt * dt;

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
    if (Uij(U, Ui_size, er_i, iiir) <= 0.0)
    {
        std::printf("Er = %e %e %e %e\n", E0, E1, Uij(U, Ui_size, er_i, iiir), dt);
        assert(false);
    }
    e = fmax(e, 0.0);
    tau[iiih] = std::pow(e, INVERSE(fgamma));
    if (Uij(U, Ui_size, er_i, iiir) <= 0.0)
    {
        //std::printf("2231242!!! %e %e %e \n", E0, Uij(U, Ui_size, er_i, iiir), dE_dt * dt);
        assert(false);
    }
    if (opts_problem == MARSHAK)
    {
        sx[iiih] = sy[iiih] = sz[iiih] = 0;
        egas[iiih] = e;
    }
}

void radiation_gpu_kernel(
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
    std::vector<double> const& rho,
    std::vector<double>& sx,
    std::vector<double>& sy,
    std::vector<double>& sz,
    std::vector<double>& egas,
    std::vector<double>& tau,
    double const fgamma,
    std::array<std::vector<double>, NRF> U,
    std::vector<double> const mmw,
    std::vector<double> const X_spc,
    std::vector<double> const Z_spc,
    double const dt,
    double const clightinv)
{
    double* d_rho{};
    cudaMalloc((void**) &d_rho, rho.size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_rho, &rho[0], rho.size(), cudaMemcpyHostToDevice));
    double* d_sx{};
    cudaMalloc((void**) &d_sx, sx.size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_sx, &sx[0], sx.size(), cudaMemcpyHostToDevice));
    double* d_sy{};
    cudaMalloc((void**) &d_sy, sy.size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_sy, &sy[0], sy.size(), cudaMemcpyHostToDevice));
    double* d_sz{};
    cudaMalloc((void**) &d_sz, sz.size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_sz, &sz[0], sz.size(), cudaMemcpyHostToDevice));
    double* d_egas{};
    cudaMalloc((void**) &d_egas, egas.size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_egas, &egas[0], egas.size(), cudaMemcpyHostToDevice));
    double* d_tau{};
    cudaMalloc((void**) &d_tau, tau.size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_tau, &tau[0], tau.size(), cudaMemcpyHostToDevice));
    double* d_U;
    cudaMalloc((void**) &d_U, NRF * U[0].size() * sizeof(double));
    for (std::size_t i = 0; i < NRF; ++i)
    {
        abort_if_cuda_error(cudaMemcpy(
            d_U + i * U.size(), &U[i][0], U[i].size(), cudaMemcpyHostToDevice));
    }
    double* d_mmw{};
    cudaMalloc((void**) &d_mmw, mmw.size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_mmw, &mmw[0], mmw.size(), cudaMemcpyHostToDevice));
    double* d_X_spc{};
    cudaMalloc((void**) &d_X_spc, X_spc.size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_X_spc, &X_spc[0], X_spc.size(), cudaMemcpyHostToDevice));
    double* d_Z_spc{};
    cudaMalloc((void**) &d_Z_spc, Z_spc.size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_Z_spc, &Z_spc[0], Z_spc.size(), cudaMemcpyHostToDevice));

    //cudaLaunchKernel(radiation_impl, 1, 1, args, 0, 0)
    radiation_impl<<<1, dim3(loop_iterations, loop_iterations, loop_iterations)>>>(
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
        d_rho, rho.size(),
        d_sx, sx.size(),
        d_sy, sy.size(),
        d_sz, sz.size(),
        d_egas, egas.size(),
        d_tau, tau.size(),
        fgamma,
        d_U, NRF * U[0].size(),
        d_mmw, mmw.size(),
        d_X_spc, X_spc.size(),
        d_Z_spc, Z_spc.size(),
        dt,
        clightinv
        );
    abort_if_cuda_error();

    abort_if_cuda_error(cudaMemcpy(
        &sx[0], d_sx, sx.size() * sizeof(double), cudaMemcpyDeviceToHost));
    abort_if_cuda_error(cudaMemcpy(
        &sy[0], d_sy, sy.size() * sizeof(double), cudaMemcpyDeviceToHost));
    abort_if_cuda_error(cudaMemcpy(
        &sz[0], d_sz, sz.size() * sizeof(double), cudaMemcpyDeviceToHost));
    abort_if_cuda_error(cudaMemcpy(&egas[0], d_egas,
        egas.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < NRF; ++i)
    {
        abort_if_cuda_error(cudaMemcpy(&U[i][0], d_U + i * U[i].size(),
            U[i].size() * sizeof(double), cudaMemcpyDeviceToHost));
    }

    abort_if_cuda_error(cudaFree(d_rho));
    abort_if_cuda_error(cudaFree(d_sx));
    abort_if_cuda_error(cudaFree(d_sy));
    abort_if_cuda_error(cudaFree(d_sz));
    abort_if_cuda_error(cudaFree(d_egas));
    abort_if_cuda_error(cudaFree(d_tau));
    abort_if_cuda_error(cudaFree(d_U));
    abort_if_cuda_error(cudaFree(d_mmw));
    abort_if_cuda_error(cudaFree(d_X_spc));
    abort_if_cuda_error(cudaFree(d_Z_spc));
}
