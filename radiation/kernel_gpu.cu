#include "config.hpp"
#include "kernel_gpu.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
    abort_if_cuda_error(cudaMemcpy(d_rho, &rho[0], rho.size(), cudaMemcpyHostToDevice));
    double* d_sx{};
    cudaMalloc((void**) &d_sx, sx.size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_sx, &sx[0], sx.size(), cudaMemcpyHostToDevice));
    double* d_sy{};
    cudaMalloc((void**) &d_sy, sy.size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_sy, &sy[0], sy.size(), cudaMemcpyHostToDevice));
    double* d_sz{};
    cudaMalloc((void**) &d_sz, sz.size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_sz, &sz[0], sz.size(), cudaMemcpyHostToDevice));
    double* d_egas{};
    cudaMalloc((void**) &d_egas, egas.size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_egas, &egas[0], egas.size(), cudaMemcpyHostToDevice));
    double* d_tau{};
    cudaMalloc((void**) &d_tau, tau.size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_tau, &tau[0], tau.size(), cudaMemcpyHostToDevice));
    double* d_U0{};
    cudaMalloc((void**) &d_U0, U[0].size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_U0, &U[0][0], U[0].size(), cudaMemcpyHostToDevice));
    double* d_U1{};
    cudaMalloc((void**) &d_U1, U[1].size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_U1, &U[1][0], U[1].size(), cudaMemcpyHostToDevice));
    double* d_U2{};
    cudaMalloc((void**) &d_U2, U[2].size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_U2, &U[2][0], U[2].size(), cudaMemcpyHostToDevice));
    double* d_U3{};
    cudaMalloc((void**) &d_U3, U[3].size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_U3, &U[3][0], U[3].size(), cudaMemcpyHostToDevice));
    double* d_mmw{};
    cudaMalloc((void**) &d_mmw, mmw.size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_mmw, &mmw[0], mmw.size(), cudaMemcpyHostToDevice));
    double* d_X_spc{};
    cudaMalloc((void**) &d_X_spc, X_spc.size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_X_spc, &X_spc[0], X_spc.size(), cudaMemcpyHostToDevice));
    double* d_Z_spc{};
    cudaMalloc((void**) &d_Z_spc, Z_spc.size() * sizeof(double));
    abort_if_cuda_error(cudaMemcpy(d_Z_spc, &Z_spc[0], Z_spc.size(), cudaMemcpyHostToDevice));
    radiation_impl<<<1, 1>>>(
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
}
