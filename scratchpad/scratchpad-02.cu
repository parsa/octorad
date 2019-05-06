#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdlib>

struct saxpy_functor
{
    double const a;

    saxpy_functor(double _a)
      : a(_a)
    {
    }

    __host__ __device__ double operator()(double const& x, double const& y) const
    {
        return a * x + y;
    }
};

void saxpy_fast(
    double A, thrust::device_vector<double>& X, thrust::device_vector<double>& Y)
{
    // Y <- A * X + Y
    thrust::transform(
        X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void saxpy_slow(
    double A, thrust::device_vector<double>& X, thrust::device_vector<double>& Y)
{
    thrust::device_vector<double> temp(X.size());

    // temp <- A
    thrust::fill(temp.begin(), temp.end(), A);

    // temp <- A * X
    thrust::transform(X.begin(), X.end(), temp.begin(), temp.begin(),
        thrust::multiplies<double>());

    // Y <- A * X + Y
    thrust::transform(
        temp.begin(), temp.end(), Y.begin(), Y.begin(), thrust::plus<double>());
}

int main()
{
    thrust::device_vector<double> a(5000000, 20.0);
    thrust::device_vector<double> b(5000000, 10.0);

    saxpy_fast(5.0f, a, b);
    return 0;
}
