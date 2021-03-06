#include "config.hpp"
#include "kernels/helpers.hpp"
#include "kernels/kernel_gpu.hpp"

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
    __device__ d_pair(T1 f, T2 s)
      : first(f)
      , second(s)
    {
    }

    T1 first{};
    T2 second{};
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

__global__ void //__launch_bounds__(512, 1)
    radiation_impl(std::int64_t const opts_eos,
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
        double const fgamma,
        double const dt,
        double const clightinv,
        payload_t* const payload_ptr)
{
    payload_t& p = *payload_ptr;

    std::int64_t const i = threadIdx.x;
    std::int64_t const j = threadIdx.y;
    std::int64_t const k = threadIdx.z + blockDim.z * blockIdx.z;

    std::int64_t const iiih = (i * grid_i_size + j) * grid_i_size + k;
    std::int64_t const iiir = (i * grid_i_size + j) * grid_i_size + k;
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
    space_vector F0 = make_space_vector(    //
        p.U[fx_i][iiir],                    //
        p.U[fy_i][iiir],                    //
        p.U[fz_i][iiir]);
    space_vector u0 = make_space_vector(vx, vy, vz);
    double E1 = E0;
    space_vector F1 = F0;
    space_vector u1 = u0;
    double e1 = e0;

    auto const ddt = implicit_radiation_step(opts_problem, physcon_c, E1, e1,
        F1, u1, den, p.mmw[iiir], p.X_spc[iiir], p.Z_spc[iiir], dt);
    double const dE_dt = ddt.first;
    double const dFx_dt = ddt.second[0];
    double const dFy_dt = ddt.second[1];
    double const dFz_dt = ddt.second[2];

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
        std::printf("Er = %e %e %e %e\n", E0, E1, p.U[er_i][iiir], dt);
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
        launch_kernel(radiation_impl,                        // kernel
            dim3(4),                                         // grid dims
            dim3(RAD_GRID_I, RAD_GRID_I, RAD_GRID_I / 4),    // block dims
            streams[stream_index],                           // stream
            opts_eos, opts_problem,                          //
            opts_dual_energy_sw1, opts_dual_energy_sw2,      //
            physcon_A, physcon_B, physcon_c,                 //
            er_i,                                            //
            fx_i, fy_i, fz_i,                                //
            d,                                               //
            RAD_GRID_I,                                      //
            GRID_ARRAY_SIZE,                                 //
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
