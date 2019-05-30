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

struct payload_t
{
    float data[1];

    __host__ __device__ float operator()(std::size_t idx) const
    {
        return data[idx];
    }

    __host__ __device__ float& operator()(std::size_t idx)
    {
        return data[idx];
    }
};

struct k1_t;

struct data_t
{
    std::promise<payload_t> promise;
    k1_t* k;
};

__global__ void k1(payload_t* payload_ptr)
{
    payload_t& payload = *payload_ptr;
    payload(0) = 3.0f;
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
        data_t& d = *d_ptr;
        // copy values over
        val()(0) = p(0);

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
        d.k = this;
        CE(cudaLaunchHostFunc(strm, callback, d_ptr));
    }

    std::future<payload_t> operator()(payload_t p)
    {
        std::promise<payload_t> prms;
        std::future<payload_t> ret = prms.get_future();
        data_t* d_ptr = new data_t;
        data_t& d = *d_ptr;
        d.promise = std::move(prms);
        launch(p, d_ptr);

        return ret;
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

    payload_t ret = *(d.k->host_payload_ptr);
    d.promise.set_value(std::move(ret));

    delete d_ptr;
}

void test()
{
    k1_t k;
    payload_t payload;

    // initialization
    payload(0) = 1.0f;

    auto f1 = k(payload);
    auto f2 = k(payload);
    payload_t outcome1 = f1.get();
    payload_t outcome2 = f2.get();

    assert(3.0f == outcome1(0));
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
