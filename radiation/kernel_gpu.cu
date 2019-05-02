#include "config.hpp"
#include "kernel_gpu.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

constexpr std::size_t lpm1 = RAD_NX - (2 * RAD_BW);

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

void abort_if_cuda_error(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::printf("cuda error: %s\n", cudaGetErrorString(err));
        std::abort();
    }
}

void abort_if_cuda_error()
{
    cudaError_t err = cudaGetLastError();
    abort_if_cuda_error(err);
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
    double* const U0, std::size_t U0_size,
    double* const U1, std::size_t U1_size,
    double* const U2, std::size_t U2_size,
    double* const U3, std::size_t U3_size,
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
    //if (opts_eos == eos_wd)
    //{
    //    e0 -= ztwd_energy(physcon_A, physcon_B, den);
    //}
    //if (e0 < egas[iiih] * opts_dual_energy_sw2)
    //{
    //    e0 = std::pow(tau[iiih], fgamma);
    //}
    //double E0 = U[er_i][iiir];
    //space_vector F0;
    //space_vector u0;
    //F0[0] = U[fx_i][iiir];
    //F0[1] = U[fy_i][iiir];
    //F0[2] = U[fz_i][iiir];
    //u0[0] = vx;
    //u0[1] = vy;
    //u0[2] = vz;
    //double E1 = E0;
    //space_vector F1 = F0;
    //space_vector u1 = u0;
    //double e1 = e0;
    //
    //auto const ddt = implicit_radiation_step(opts_problem, physcon_c, E1, e1,
    //    F1, u1, den, mmw[iiir], X_spc[iiir], Z_spc[iiir], dt);
    //double const dE_dt = ddt.first;
    //double const dFx_dt = ddt.second[0];
    //double const dFy_dt = ddt.second[1];
    //double const dFz_dt = ddt.second[2];
    //
    //// Accumulate derivatives
    //U[er_i][iiir] += dE_dt * dt;
    //U[fx_i][iiir] += dFx_dt * dt;
    //U[fy_i][iiir] += dFy_dt * dt;
    //U[fz_i][iiir] += dFz_dt * dt;
    //
    //egas[iiih] -= dE_dt * dt;
    //sx[iiih] -= dFx_dt * dt * clightinv * clightinv;
    //sy[iiih] -= dFy_dt * dt * clightinv * clightinv;
    //sz[iiih] -= dFz_dt * dt * clightinv * clightinv;
    //
    //// Find tau with dual energy formalism
    //double e = egas[iiih];
    //e -= 0.5 * sx[iiih] * sx[iiih] * deninv;
    //e -= 0.5 * sy[iiih] * sy[iiih] * deninv;
    //e -= 0.5 * sz[iiih] * sz[iiih] * deninv;
    //if (opts_eos == eos_wd)
    //{
    //    e -= ztwd_energy(physcon_A, physcon_B, den);
    //}
    //if (e < opts_dual_energy_sw1 * egas[iiih])
    //{
    //    e = e1;
    //}
    //if (opts_problem == MARSHAK)
    //{
    //    egas[iiih] = e;
    //    sx[iiih] = sy[iiih] = sz[iiih] = 0;
    //}
    //if (U[er_i][iiir] <= 0.0)
    //{
    //    std::printf("Er = %e %e %e %e\n", E0, E1, U[er_i][iiir], dt);
    //    std::abort();
    //}
    //e = std::max(e, 0.0);
    //tau[iiih] = std::pow(e, INVERSE(fgamma));
    //if (U[er_i][iiir] <= 0.0)
    //{
    //    std::printf("2231242!!! %e %e %e \n", E0, U[er_i][iiir], dE_dt * dt);
    //    std::abort();
    //}
    //if (opts_problem == MARSHAK)
    //{
    //    sx[iiih] = sy[iiih] = sz[iiih] = 0;
    //    egas[iiih] = e;
    //}
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
    double* d_U0{};
    cudaMalloc((void**) &d_U0, U[0].size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_U0, &U[0][0], U[0].size(), cudaMemcpyHostToDevice));
    double* d_U1{};
    cudaMalloc((void**) &d_U1, U[1].size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_U1, &U[1][0], U[1].size(), cudaMemcpyHostToDevice));
    double* d_U2{};
    cudaMalloc((void**) &d_U2, U[2].size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_U2, &U[2][0], U[2].size(), cudaMemcpyHostToDevice));
    double* d_U3{};
    cudaMalloc((void**) &d_U3, U[3].size() * sizeof(double));
    abort_if_cuda_error(
        cudaMemcpy(d_U3, &U[3][0], U[3].size(), cudaMemcpyHostToDevice));
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
    radiation_impl<<<1, dim3(lpm1, lpm1, lpm1)>>>(
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
        d_U0, U[0].size(),
        d_U1, U[1].size(),
        d_U2, U[2].size(),
        d_U3, U[3].size(),
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
    abort_if_cuda_error(cudaMemcpy(
        &U[0][0], d_U0, U[0].size() * sizeof(double), cudaMemcpyDeviceToHost));
    abort_if_cuda_error(cudaMemcpy(
        &U[1][0], d_U0, U[1].size() * sizeof(double), cudaMemcpyDeviceToHost));
    abort_if_cuda_error(cudaMemcpy(
        &U[2][0], d_U0, U[2].size() * sizeof(double), cudaMemcpyDeviceToHost));
    abort_if_cuda_error(cudaMemcpy(
        &U[3][0], d_U0, U[3].size() * sizeof(double), cudaMemcpyDeviceToHost));

    abort_if_cuda_error(cudaFree(d_rho));
    abort_if_cuda_error(cudaFree(d_sx));
    abort_if_cuda_error(cudaFree(d_sy));
    abort_if_cuda_error(cudaFree(d_sz));
    abort_if_cuda_error(cudaFree(d_egas));
    abort_if_cuda_error(cudaFree(d_tau));
    abort_if_cuda_error(cudaFree(d_U0));
    abort_if_cuda_error(cudaFree(d_U1));
    abort_if_cuda_error(cudaFree(d_U2));
    abort_if_cuda_error(cudaFree(d_U3));
    abort_if_cuda_error(cudaFree(d_mmw));
    abort_if_cuda_error(cudaFree(d_X_spc));
    abort_if_cuda_error(cudaFree(d_Z_spc));
}
