#include <cuda_runtime.h>

#include <cassert>
#include <exception>
#include <future>
#include <iostream>
#include <string>

#define CE(err)                                                                \
    {                                                                          \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            std::cout << "CUDA error in " << __FUNCTION__ << " (" << __FILE__  \
                      << ":" << __LINE__ << ") - " << cudaGetErrorString(err); \
            std::terminate();                                                  \
        }                                                                      \
    }

constexpr std::size_t data_COUNT = 1;

struct payload_t
{
    float data[data_COUNT];

    float* begin()
    {
        return &data[0];
    }

    float* end()
    {
        return &data[0] + data_COUNT;
    }

    float const* begin() const
    {
        return &data[0];
    }

    float const* end() const
    {
        return &data[0] + data_COUNT;
    }

    std::size_t size() const
    {
        return data_COUNT;
    }

    __host__ __device__ float operator[](std::size_t idx) const
    {
        return data[idx];
    }

    __host__ __device__ float& operator[](std::size_t idx)
    {
        return data[idx];
    }
};

struct k1_t;

struct data_t
{
    data_t(std::promise<payload_t> p, k1_t& k_)
      : promise(std::move(p))
      , kernel(k_)
    {
    }
    std::promise<payload_t> promise;
    k1_t& kernel;
};

__global__ void k1(payload_t* payload_ptr)
{
    payload_t& payload = *payload_ptr;
    payload[0] = 3.0f;
}

void CUDART_CB callback(void* args);

struct k1_t
{
    k1_t()
    {
        CE(cudaStreamCreate(&strm));
        CE(cudaMallocHost(&host_payload_ptr, sizeof(payload_t)));
        CE(cudaMalloc(&dev_payload_ptr, sizeof(payload_t)));
    }
    ~k1_t()
    {
        // ensure all operations in the stream are done
        CE(cudaStreamSynchronize(strm));
        // free device and then pinned memory
        CE(cudaFree(dev_payload_ptr));
        CE(cudaFreeHost(host_payload_ptr));
        // destroy stream
        CE(cudaStreamDestroy(strm));
    }

    void launch(payload_t const& p, data_t* d_ptr)
    {
        // copy values over
        payload_t& v = val();
        std::copy(p.begin(), p.end(), v.begin());
        val()[0] = p[0];

        // copy to device
        CE(cudaMemcpyAsync(dev_payload_ptr, host_payload_ptr, sizeof(payload_t),
            cudaMemcpyHostToDevice, strm));
        // launch kernel
        {
            void* args[] = {&dev_payload_ptr};
            CE(cudaLaunchKernel(k1, dim3(1), dim3(1), args, 0, strm));
        }
        // copy back
        CE(cudaMemcpyAsync(host_payload_ptr, dev_payload_ptr, sizeof(payload_t),
            cudaMemcpyDeviceToHost, strm));
        // async callback
        CE(cudaLaunchHostFunc(strm, callback, d_ptr));
    }

    std::future<payload_t> operator()(payload_t& p)
    {
        std::promise<payload_t> payload_promise;
        std::future<payload_t> payload_future = payload_promise.get_future();
        data_t* d_ptr = new data_t(std::move(payload_promise), *this);
        launch(p, d_ptr);

        return payload_future;
    }

    payload_t val() const
    {
        return *host_payload_ptr;
    }

    payload_t& val()
    {
        return *host_payload_ptr;
    }

    cudaStream_t strm;
    payload_t* host_payload_ptr = nullptr;
    payload_t* dev_payload_ptr = nullptr;
};

void CUDART_CB callback(void* args)
{
    data_t* d_ptr = static_cast<data_t*>(args);
    data_t& d = *d_ptr;

    payload_t ret = *(d.kernel.host_payload_ptr);
    d.promise.set_value(std::move(ret));

    delete d_ptr;
}

void test()
{
    k1_t k;
    payload_t payload;

    // initialization
    payload[0] = 1.0f;

    auto f1 = k(payload);
    auto f2 = k(payload);
    payload_t outcome1 = f1.get();
    payload_t outcome2 = f2.get();

    assert(3.0f == outcome1[0]);
}

int main()
{
    try
    {
        test();
    }
    catch (std::exception const& ex)
    {
        std::cout << "exception: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
