#include "config.hpp"
#include "kernels/helpers.hpp"
#include "kernels/kernel_gpu.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel_gpu.hpp"

using namespace octotiger::detail;

///////////////////////////////////////////////////////////////////////////////
// stream management
///////////////////////////////////////////////////////////////////////////////
// these exist so that cudaStream_t is not exposed through the API
// otherwise a cudaStream_t can be put directly in the kernel struct
std::mutex m;
std::size_t stream_top = 0;
std::map<std::size_t, cudaStream_t> streams;

///////////////////////////////////////////////////////////////////////////////
// device structs
///////////////////////////////////////////////////////////////////////////////
struct space_vector
{
    __host__ __device__ space_vector()
      : x(0.0)
      , y(0.0)
      , z(0.0)
    {
    }
    __host__ __device__ space_vector(double vx, double vy, double vz)
      : x(vx)
      , y(vy)
      , z(vz)
    {
    }

    __host__ __device__ double& operator[](std::size_t const index)
    {
        switch (index)
        {
        case 0:
            return x;
        case 1:
            return y;
        default:
            return z;
        }
    }

    __host__ __device__ double operator[](std::size_t const index) const
    {
        switch (index)
        {
        case 0:
            return x;
        case 1:
            return y;
        default:
            return z;
        }
    }

    __device__ void scale(double const factor)
    {
        x *= factor;
        y *= factor;
        z *= factor;
    }

    __device__ double dot(space_vector const& rhs) const
    {
        return (x * rhs.x) + (y * rhs.y) + (z * rhs.z);
    }

    double x{};
    double y{};
    double z{};
};

__device__ inline space_vector make_space_vector(double x, double y, double z)
{
    return space_vector{x, y, z};
}

struct radiation_step_result
{
    double dE_dt{};
    space_vector dF_dt{};
    double final_e{};
};

///////////////////////////////////////////////////////////////////////////////
// misc fx
///////////////////////////////////////////////////////////////////////////////
inline std::int64_t hindex(std::int64_t i, std::int64_t j, std::int64_t k)
{
    return i * H_DNX + j * H_DNY + k * H_DNZ;
}

__device__ inline double INVERSE(double a)
{
    return 1.0 / a;
}

// HACK: only implemented for marshak
template <typename T>
__device__ T B_p(
    std::int64_t const opts_problem, double const physcon_c, T rho, T e, T mmw)
{
    assert(opts_problem == MARSHAK);
    return T((physcon_c / 4.0 / M_PI)) * e;
}

// HACK: only implemented for marshak
template <typename T>
__device__ T kappa_p(
    std::int64_t const opts_problem, T rho, T e, T mmw, double X, double Z)
{
    assert(opts_problem == MARSHAK);
    return MARSHAK_OPAC;
}

// HACK: only implemented for marshak
template <typename T>
__device__ T kappa_R(
    std::int64_t const opts_problem, T rho, T e, T mmw, double X, double Z)
{
    assert(opts_problem == MARSHAK);
    return MARSHAK_OPAC;
}

// square of a number
template <typename T>
__device__ inline T sqr(T const& val)
{
    return val * val;
}

// cube of a number
template <typename T>
__device__ constexpr inline T cube(T const& val)
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

///////////////////////////////////////////////////////////////////////////////
// kernel implementation
///////////////////////////////////////////////////////////////////////////////
template <typename Fx>
__device__ void abort_if_solver_not_converged(double const eg_t0, double E0,
    Fx const test, double const E, double const eg_t)
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
    constexpr std::size_t MAX_ITERATIONS = 50;
    // Errors
    double const error_tolerance = 1.0e-9;

    for (std::size_t i = 0; i < MAX_ITERATIONS; ++i)
    {
        // Root solver has converged if error is smaller that error tolerance
        double const error = fmax(fabs(f_mid), fabs(f_min)) / (E + eg_t);
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

__noinline__ __device__ radiation_step_result implicit_radiation_step(
    std::int64_t const opts_problem, double const physcon_c, double E0,
    double& e0, space_vector F0, space_vector u0, double const rho,
    double const mmw, double const X, double const Z, double const dt)
{
    double const c = physcon_c;
    double const rhoc2 = rho * c * c;
    double const inv_rhoc2 = 1.0 / rhoc2;
    double const inv_c = 1.0 / c;
    double const inv_rhoc2_c = inv_rhoc2 * inv_c;
    double const dt_c = dt * c;

    double kp = kappa_p(opts_problem, rho, e0, mmw, X, Z);
    double kr = kappa_R(opts_problem, rho, e0, mmw, X, Z);
    kp *= dt_c;
    kr *= dt_c;

    double const E0_n = E0 * inv_rhoc2;
    double const F0x = F0.x * inv_rhoc2_c;
    double const F0y = F0.y * inv_rhoc2_c;
    double const F0z = F0.z * inv_rhoc2_c;
    double const u0x = u0.x * inv_c;
    double const u0y = u0.y * inv_c;
    double const u0z = u0.z * inv_c;

    double const e0_n = e0 * inv_rhoc2;
    double const eg_t0 = e0_n + 0.5 * (u0x * u0x + u0y * u0y + u0z * u0z);
    double const B_scale = (4.0 * M_PI / c) * inv_rhoc2;

    auto evaluate = [&](double de, double& E_out, double& eg_t_out,
                        double& ei_out, double& Fx_out, double& Fy_out,
                        double& Fz_out) -> double {
        E_out = E0_n + de;
        double const denom = 1.0 + kr * (1.0 + (4.0 / 3.0) * E_out);
        double const denom_inv = 1.0 / denom;
        double const k4E = (4.0 / 3.0) * kr * E_out;

        double u2 = 0.0;
        double udotF = 0.0;

        double const Fx_base = F0x;
        double const Fx_term = k4E * (u0x + Fx_base);
        Fx_out = (Fx_base + Fx_term) * denom_inv;
        double const ux = u0x + Fx_base - Fx_out;
        u2 += ux * ux;
        udotF += Fx_out * ux;

        double const Fy_base = F0y;
        double const Fy_term = k4E * (u0y + Fy_base);
        Fy_out = (Fy_base + Fy_term) * denom_inv;
        double const uy = u0y + Fy_base - Fy_out;
        u2 += uy * uy;
        udotF += Fy_out * uy;

        double const Fz_base = F0z;
        double const Fz_term = k4E * (u0z + Fz_base);
        Fz_out = (Fz_base + Fz_term) * denom_inv;
        double const uz = u0z + Fz_base - Fz_out;
        u2 += uz * uz;
        udotF += Fz_out * uz;

        ei_out = fmax(eg_t0 - E_out + E0_n - 0.5 * u2, 0.0);
        double const b =
            B_scale * B_p(opts_problem, physcon_c, rho, ei_out * rhoc2, mmw);
        eg_t_out = eg_t0 + E0_n - E_out;
        return (E_out - E0_n) + kp * (E_out - b) + (kr - 2.0 * kp) * udotF;
    };

    double de_max = eg_t0;
    double de_mid = 0.0;
    double de_min = -E0_n;

    double E_mid = 0.0;
    double eg_t_mid = 0.0;
    double ei_mid = 0.0;
    double Fx_mid = F0x;
    double Fy_mid = F0y;
    double Fz_mid = F0z;
    double f_mid =
        evaluate(de_mid, E_mid, eg_t_mid, ei_mid, Fx_mid, Fy_mid, Fz_mid);

    double f_min = 0.0;
    {
        double E_tmp = 0.0;
        double eg_t_tmp = 0.0;
        double ei_tmp = 0.0;
        double Fx_tmp = F0x;
        double Fy_tmp = F0y;
        double Fz_tmp = F0z;
        f_min = evaluate(de_min, E_tmp, eg_t_tmp, ei_tmp, Fx_tmp, Fy_tmp, Fz_tmp);
    }

    constexpr std::size_t MAX_ITERATIONS = 50;
    double const error_tolerance = 1.0e-9;

    for (std::size_t iter = 0; iter < MAX_ITERATIONS; ++iter)
    {
        #if defined(OCTORAD_RELAXED)
        double const denom_err = fabs(E_mid) + fabs(eg_t_mid) + 1e-12;
        #else
        double const denom_err = E_mid + eg_t_mid;
        #endif
        double const error = fmax(fabs(f_mid), fabs(f_min)) / denom_err;
        if (error < error_tolerance)
        {
            break;
        }

        if ((f_min < 0.0) != (f_mid < 0.0))
        {
            de_max = de_mid;
            de_mid = 0.5 * (de_min + de_max);
        }
        else
        {
            de_min = de_mid;
            f_min = f_mid;
            de_mid = 0.5 * (de_min + de_max);
        }

        f_mid =
            evaluate(de_mid, E_mid, eg_t_mid, ei_mid, Fx_mid, Fy_mid, Fz_mid);
    }

    #if defined(OCTORAD_RELAXED)
    double const final_error =
        fmax(fabs(f_mid), fabs(f_min)) / (fabs(E_mid) + fabs(eg_t_mid) + 1e-12);
    (void)final_error;
    #else
    double const final_error =
        fmax(fabs(f_mid), fabs(f_min)) / (E_mid + eg_t_mid);
    if (final_error >= error_tolerance)
    {
        printf("implicit radiation solver failed to converge: err=%e\n",
            final_error);
        assert(false);
    }
    #endif

    double const dtinv = 1.0 / dt;
    radiation_step_result result;
    result.dE_dt = (E_mid - E0_n) * dtinv * rhoc2;
    result.dF_dt = make_space_vector(
        (Fx_mid - F0x) * dtinv * rhoc2 * c,
        (Fy_mid - F0y) * dtinv * rhoc2 * c,
        (Fz_mid - F0z) * dtinv * rhoc2 * c);
    result.final_e = ei_mid * rhoc2;

    return result;
}

__device__ inline int lindex(int i, int j, int k, int n)
{
    return (i * n + j) * n + k;
}

#if defined(OCTORAD_RELAXED)
__global__ __launch_bounds__(128, 2)
#else
__global__ __launch_bounds__(128, 5)
#endif
void radiation_impl(std::int64_t const opts_eos,
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
        int const grid_i_size,
        int const Ui_size,
    double const fgamma,
    double const dt,
    double const clightinv,
    payload_t* const payload_ptr)
{
    payload_t& p = *payload_ptr;

    int const i = threadIdx.x + blockDim.x * blockIdx.x;
    int const j = threadIdx.y + blockDim.y * blockIdx.y;
    int const k = threadIdx.z + blockDim.z * blockIdx.z;
    if (i >= grid_i_size || j >= grid_i_size || k >= grid_i_size)
    {
        return;
    }

    int const iiih = lindex(i, j, k, grid_i_size);
    int const iiir = iiih;
    double const den = p.rho[iiih];
    double const deninv = INVERSE(den);
    double vx = p.sx[iiih] * deninv;
    double vy = p.sy[iiih] * deninv;
    double vz = p.sz[iiih] * deninv;

    // Compute e0 from dual energy formalism
    double e0 = p.egas[iiih]     //
        - 0.5 * vx * vx * den    //
        - 0.5 * vy * vy * den    //
        - 0.5 * vz * vz * den;
    if (opts_eos == eos_wd)
    {
        e0 -= ztwd_energy(physcon_A, physcon_B, den);
    }
    if (e0 < p.egas[iiih] * opts_dual_energy_sw2)
    {
        e0 = pow(p.tau[iiih], fgamma);
    }
    double E0 = p.U[er_i][iiir];
    space_vector F0 = make_space_vector(
        p.U[fx_i][iiir], p.U[fy_i][iiir], p.U[fz_i][iiir]);
    space_vector u0 = make_space_vector(vx, vy, vz);

    auto const step = implicit_radiation_step(opts_problem, physcon_c, E0, e0,
        F0, u0, den, p.mmw[iiir], p.X_spc[iiir], p.Z_spc[iiir], dt);
    double const dE_dt = step.dE_dt;
    double const dFx_dt = step.dF_dt[0];
    double const dFy_dt = step.dF_dt[1];
    double const dFz_dt = step.dF_dt[2];
    double const updated_E = E0 + dE_dt * dt;
    double const e1 = step.final_e;

    // Accumulate derivatives
    p.U[er_i][iiir] += dE_dt * dt;
    p.U[fx_i][iiir] += dFx_dt * dt;
    p.U[fy_i][iiir] += dFy_dt * dt;
    p.U[fz_i][iiir] += dFz_dt * dt;

    p.egas[iiih] -= dE_dt * dt;
    p.sx[iiih] -= dFx_dt * dt * clightinv * clightinv;
    p.sy[iiih] -= dFy_dt * dt * clightinv * clightinv;
    p.sz[iiih] -= dFz_dt * dt * clightinv * clightinv;

    // Find tau with dual energy formalism
    double e = p.egas[iiih]                         //
        - 0.5 * p.sx[iiih] * p.sx[iiih] * deninv    //
        - 0.5 * p.sy[iiih] * p.sy[iiih] * deninv    //
        - 0.5 * p.sz[iiih] * p.sz[iiih] * deninv;
    if (opts_eos == eos_wd)
    {
        e -= ztwd_energy(physcon_A, physcon_B, den);
    }
    if (e < opts_dual_energy_sw1 * p.egas[iiih])
    {
        e = e1;
    }
    if (p.U[er_i][iiir] <= 0.0)
    {
        std::printf("Er = %e %e %e %e\n", E0, updated_E, p.U[er_i][iiir], dt);
        assert(false);
    }
    e = fmax(e, 0.0);
    p.tau[iiih] = std::pow(e, INVERSE(fgamma));
    if (p.U[er_i][iiir] <= 0.0)
    {
        std::printf("2231242!!! %e %e %e \n", E0, p.U[er_i][iiir], dE_dt * dt);
        assert(false);
    }
    if (opts_problem == MARSHAK)
    {
        p.egas[iiih] = e;
        p.sx[iiih] = 0.0;
        p.sy[iiih] = 0.0;
        p.sz[iiih] = 0.0;
    }
}

namespace octotiger {
    void radiation_gpu_kernel::enqueue(std::int64_t const opts_eos,
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
        std::array<std::vector<double>, NRF>& U,
        std::vector<double> const& mmw,
        std::vector<double> const& X_spc,
        std::vector<double> const& Z_spc,
        double const dt,
        double const clightinv,
        bool const copy_back)
    {
        payload_t& payload = *h_payload_ptr;
        load_args(payload, d, sx, sy, sz, egas, tau, U, rho, X_spc, Z_spc, mmw);
        device_copy_from_host(copy_policy::async, d_payload_ptr.get(),
            h_payload_ptr.get(), streams[stream_index]);

        std::size_t const grid_i = RAD_GRID_I;
        unsigned const bx = 8;
        unsigned const by = 8;
        unsigned const bz = 2;
        dim3 const block_dim(bx, by, bz);
        dim3 const grid_dim(
            static_cast<unsigned>((grid_i + bx - 1) / bx),
            static_cast<unsigned>((grid_i + by - 1) / by),
            static_cast<unsigned>((grid_i + bz - 1) / bz));

        launch_kernel(radiation_impl, grid_dim, block_dim, streams[stream_index],
            opts_eos, opts_problem, opts_dual_energy_sw1, opts_dual_energy_sw2,
            physcon_A, physcon_B, physcon_c, er_i, fx_i, fy_i, fz_i, d,
            static_cast<int>(RAD_GRID_I), static_cast<int>(GRID_ARRAY_SIZE),
            fgamma, dt, clightinv, d_payload_ptr.get());

        if (copy_back)
        {
            device_copy_to_host<output_payload_t>(copy_policy::async,
                d_payload_ptr.get(), h_payload_ptr.get(), streams[stream_index]);
        }
    }

    void radiation_gpu_kernel::stream_sync()
    {
        device_stream_sync(streams[stream_index]);
    }
}

///////////////////////////////////////////////////////////////////////////////
// wrapper implementation
///////////////////////////////////////////////////////////////////////////////
namespace octotiger {
    namespace detail {
        void host_payload_deleter::operator()(payload_t* ptr)
        {
            host_pinned_free(ptr);
        }

        void device_payload_deleter::operator()(payload_t* ptr)
        {
            device_free(ptr);
        }
    }

    void device_init()
    {
        throw_if_cuda_error(cudaFree(0));
    }

    void device_reset()
    {
        throw_if_cuda_error(cudaDeviceReset());
    }

    radiation_gpu_kernel::radiation_gpu_kernel()
      : d_payload_ptr(device_alloc<payload_t>())
      , h_payload_ptr(host_pinned_alloc<payload_t>())
    {
        std::lock_guard<std::mutex> l(m);
        stream_index = stream_top++;
        streams[stream_index] = device_stream_create();
    }

    radiation_gpu_kernel::radiation_gpu_kernel(radiation_gpu_kernel&& other)
    {
        d_payload_ptr = std::move(other.d_payload_ptr);
        h_payload_ptr = std::move(other.h_payload_ptr);
        std::swap(stream_index, other.stream_index);
        other.moved = true;
    }

    radiation_gpu_kernel& radiation_gpu_kernel::operator=(
        radiation_gpu_kernel&& other)
    {
        d_payload_ptr = std::move(other.d_payload_ptr);
        h_payload_ptr = std::move(other.h_payload_ptr);
        std::swap(stream_index, other.stream_index);
        other.moved = true;

        return *this;
    }

    radiation_gpu_kernel::~radiation_gpu_kernel()
    {
        if (!moved)
        {
            device_stream_destroy(streams[stream_index]);
            {
                std::lock_guard<std::mutex> l(m);
                streams.erase(stream_index);
            }
        }
    }

    void radiation_gpu_kernel::load_args(payload_t& payload,
        std::int64_t const d, std::vector<double>& sx, std::vector<double>& sy,
        std::vector<double>& sz, std::vector<double>& egas,
        std::vector<double>& tau, std::array<std::vector<double>, NRF>& U,
        std::vector<double> const& rho, std::vector<double> const& X_spc,
        std::vector<double> const& Z_spc, std::vector<double> const& mmw)
    {
        std::size_t index_counter = 0;
        for (std::size_t i = RAD_BW; i != RAD_NX - RAD_BW; ++i)
        {
            for (std::size_t j = RAD_BW; j != RAD_NX - RAD_BW; ++j)
            {
                for (std::size_t k = RAD_BW; k != RAD_NX - RAD_BW; ++k)
                {
                    // padded index
                    std::size_t const iiih = hindex(i + d, j + d, k + d);
                    // copy output arrays sx, sy, sz, egas, tau, U[0:NRF - 1]
                    payload.sx[index_counter] = sx[iiih];
                    payload.sy[index_counter] = sy[iiih];
                    payload.sz[index_counter] = sz[iiih];
                    payload.egas[index_counter] = egas[iiih];
                    payload.tau[index_counter] = tau[iiih];
                    for (std::size_t l = 0; l < NRF; ++l)
                    {
                        payload.U[l][index_counter] = U[l][iiih];
                    }
                    // copy input arrays rho, X_spc, Z_spc, mmw
                    payload.rho[index_counter] = rho[iiih];
                    payload.X_spc[index_counter] = X_spc[iiih];
                    payload.Z_spc[index_counter] = Z_spc[iiih];
                    payload.mmw[index_counter] = mmw[iiih];

                    ++index_counter;
                }
            }
        }
    }

    void radiation_gpu_kernel::update_outputs(payload_t& payload,
        std::int64_t const d, std::vector<double>& sx, std::vector<double>& sy,
        std::vector<double>& sz, std::vector<double>& egas,
        std::vector<double>& tau, std::array<std::vector<double>, NRF>& U)
    {
        std::size_t index_counter{};
        for (std::size_t i = RAD_BW; i < RAD_NX - RAD_BW; ++i)
        {
            for (std::size_t j = RAD_BW; j < RAD_NX - RAD_BW; ++j)
            {
                for (std::size_t k = RAD_BW; k < RAD_NX - RAD_BW; ++k)
                {
                    // padded index
                    std::size_t const iiih = hindex(i + d, j + d, k + d);
                    // update output arrays sx, sy, sz, egas, U[0:NRF - 1]
                    sx[iiih] = payload.sx[index_counter];
                    sy[iiih] = payload.sy[index_counter];
                    sz[iiih] = payload.sz[index_counter];
                    egas[iiih] = payload.egas[index_counter];
                    tau[iiih] = payload.tau[index_counter];
                    for (std::size_t l = 0; l < NRF; ++l)
                    {
                        U[l][iiih] = payload.U[l][index_counter];
                    }

                    ++index_counter;
                }
            }
        }
    }

    void radiation_gpu_kernel::operator()(std::int64_t const opts_eos,
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
        std::array<std::vector<double>, NRF>& U,
        std::vector<double> const& mmw,
        std::vector<double> const& X_spc,
        std::vector<double> const& Z_spc,
        double const dt,
        double const clightinv)
    {
        // avoid working directly with the pointer syntax when possible
        payload_t& payload = *h_payload_ptr;

        // only extract data that is needed by the kernel
        load_args(payload, d, sx, sy, sz, egas, tau, U, rho, X_spc, Z_spc, mmw);
        // memcpy array args to the gpu
        device_copy_from_host(copy_policy::async, d_payload_ptr.get(),
            h_payload_ptr.get(), streams[stream_index]);

        // launch the kernel
        std::size_t const grid_i = RAD_GRID_I;
        unsigned const bx = 8;
        unsigned const by = 8;
        unsigned const bz = 2;
        dim3 const block_dim(bx, by, bz);
        dim3 const grid_dim(
            static_cast<unsigned>((grid_i + bx - 1) / bx),
            static_cast<unsigned>((grid_i + by - 1) / by),
            static_cast<unsigned>((grid_i + bz - 1) / bz));

        launch_kernel(radiation_impl,                        // kernel
            grid_dim,                                        // grid dims
            block_dim,                                       // block dims
            streams[stream_index],                           // stream
            opts_eos, opts_problem,                          //
            opts_dual_energy_sw1, opts_dual_energy_sw2,      //
            physcon_A, physcon_B, physcon_c,                 //
            er_i,                                            //
            fx_i, fy_i, fz_i,                                //
            d,                                               //
            static_cast<int>(RAD_GRID_I),                    //
            static_cast<int>(GRID_ARRAY_SIZE),               //
            fgamma,                                          //
            dt,                                              //
            clightinv,                                       //
            d_payload_ptr.get()                              //
        );

        // memcpy output arrays from gpu
        // only overwrite the output portion
        // NOTE: payload memory layout begins with output arrays
        device_copy_to_host<output_payload_t>(copy_policy::async,
            d_payload_ptr.get(), h_payload_ptr.get(), streams[stream_index]);
        // barrier. wait until kernel finishes execution
        device_stream_sync(streams[stream_index]);

        // write updated data back
        update_outputs(payload, d, sx, sy, sz, egas, tau, U);
    }
}
