#include <cuda_runtime.h>

#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

namespace pcw {
#define CE(err)                                                                \
    {                                                                          \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            std::stringstream err_ss;                                          \
            err_ss << "CUDA error in " << __FUNCTION__ << " (" << __FILE__     \
                   << ":" << __LINE__ << ") - " << cudaGetErrorString(err);    \
            throw std::runtime_error(err_ss.str());                            \
        }                                                                      \
    }

    std::size_t device_count()
    {
        int ret = 0;
        CE(cudaGetDeviceCount(&ret));
        return static_cast<std::size_t>(ret);
    }
    std::size_t current_device_id()
    {
        int ret = 0;
        CE(cudaGetDevice(&ret));
        return static_cast<std::size_t>(ret);
    }
    cudaDeviceProp device_props(std::size_t id)
    {
        cudaDeviceProp ret;
        CE(
            cudaGetDeviceProperties(&ret, static_cast<int>(id)));
        return (ret);
    }
    void set_current_device(std::size_t id)
    {
        CE(cudaSetDevice(static_cast<int>(id)));
    }
    std::pair<int, int> get_driver_version(std::size_t id)
    {
        int ret = 0;
        CE(cudaDriverGetVersion(&ret));
        return std::make_pair(ret / 1000, (ret % 100) / 10);
    }
    std::pair<int, int> get_runtime_version(std::size_t id)
    {
        int ret = 0;
        CE(cudaRuntimeGetVersion(&ret));
        return std::make_pair(ret / 1000, (ret % 100) / 10);
    }
}

std::map<int, std::string> const compute_mode_map{
    {cudaComputeModeDefault, "Default"},
    {cudaComputeModeExclusive, "Exclusive"},
    {cudaComputeModeProhibited, "Prohibited"},
    {cudaComputeModeExclusiveProcess, "ExclusiveProcess"},
};

void print_device_info(std::size_t device_id)
{
    cudaDeviceProp prop = pcw::device_props(device_id);
    std::cout << std::boolalpha
        << "==================== Device " << device_id << " ====================\n"
        << "Name: " << prop.name << '\n'
        << "Clock Rate: " << (prop.clockRate / 1e6) << " GHz\n"
        << "Multi Processor Count: " << prop.multiProcessorCount << '\n'
        << "Async Engine Count: " << prop.asyncEngineCount << '\n'
        << "Concurrent Kernels: " << prop.concurrentKernels << '\n'
        << "Compute Capability: " << prop.major << '.' << prop.minor << '\n'
        << "LUID Device Node Mask: " << prop.luidDeviceNodeMask << '\n'
        << "Total Global Memory: " << (prop.totalGlobalMem >> 20) << " MB\n"
        << "Total Constant Memory: " << (prop.totalConstMem >> 10) << " KB\n"
        << "Shared Memory Per Block: " << (prop.sharedMemPerBlock >> 10) << " KB\n"
        << "Shared Memory Per Block Optin: " << (prop.sharedMemPerBlockOptin >> 10) << " KB\n"
        << "Shared Memory Per Multiprocessor: " << (prop.sharedMemPerMultiprocessor >> 10) << " KB\n"
        << "Registers Per Block: " << prop.regsPerBlock << '\n'
        << "Registers Per Multiprocessor: " << prop.regsPerMultiprocessor << '\n'
        << "Warp Size: " << prop.warpSize << '\n'
        << "Max Threads Per Multi Processor: " << prop.maxThreadsPerMultiProcessor << '\n'
        << "MemoryPitch: " << prop.memPitch << '\n'
        << "Max Threads Per Block: " << prop.maxThreadsPerBlock << '\n'
        << "Max Threads Dimensions: (" << prop.maxThreadsDim[0]
        << ", " << prop.maxThreadsDim[1]
        << ", " << prop.maxThreadsDim[2] << ")\n"
        << "Max Grid Size: (" << prop.maxGridSize[0]
        << ", " << prop.maxGridSize[1]
        << ", " << prop.maxGridSize[2] << ")\n"
        << "Texture Alignment: " << prop.textureAlignment << '\n'
        << "Texture Pitch Alignment: " << prop.texturePitchAlignment << '\n'
        << "Kernel Exec Timeout Enabled: " << (bool)prop.kernelExecTimeoutEnabled << '\n'
        << "Integrated: " << (bool)prop.integrated << '\n'
        << "Can Map Host Memory: " << (bool)prop.canMapHostMemory << '\n'
        << "Compute Mode: " << compute_mode_map.at(prop.computeMode) << '\n'
        << "ECC Enabled: " << (bool)prop.ECCEnabled << '\n'
        << "PCI BusID: " << prop.pciBusID << '\n'
        << "PCI DeviceID: " << prop.pciDeviceID << '\n'
        << "PCI DomainID: " << prop.pciDomainID << '\n'
        << "Is Tesla and uses TCC Driver: " << (bool)prop.tccDriver << '\n'
        << "Memory Clock Rate: " << (prop.memoryClockRate / 1e6) << " GHz\n"
        << "Memory Bus Width: " << prop.memoryBusWidth << "\n"
        << "L2 Cache Size: " << (prop.l2CacheSize >> 20) << " MB\n"
        << "Global L1 Cache Supported: " << (bool)prop.globalL1CacheSupported << '\n'
        << "Local L1 Cache Supported: " << (bool)prop.localL1CacheSupported << '\n'
        << "Unified Addressing: " << (bool)prop.unifiedAddressing << '\n'
        << "Managed Memory: " << (bool)prop.managedMemory << '\n'
        << "Pageable Memory Access: " << (bool)prop.pageableMemoryAccess << '\n'
        << "Pageable Memory Access Uses Host Page Tables: " << (bool)prop.pageableMemoryAccessUsesHostPageTables << '\n'
        << "Direct Managed Memory Access From Host: " << (bool)prop.directManagedMemAccessFromHost << '\n'
        << "Stream Priorities Supported: " << (bool)prop.streamPrioritiesSupported << '\n'
        << "Is Multi GPU Board: " << (bool)prop.isMultiGpuBoard << '\n'
        << "Multi GPU Board Group ID: " << prop.multiGpuBoardGroupID << '\n'
        << "Host Native Atomic Supported: " << (bool)prop.hostNativeAtomicSupported << '\n'
        << "Single To Double Precision Performance Ratio: " << prop.singleToDoublePrecisionPerfRatio << '\n'
        << "Concurrent Managed Access: " << (bool)prop.concurrentManagedAccess << '\n'
        << "Compute Preemption Supported: " << (bool)prop.computePreemptionSupported << '\n'
        << "Can Use Host Pointer For Registered Memory: " << (bool)prop.canUseHostPointerForRegisteredMem << '\n'
        << "Cooperative Launch: " << (bool)prop.cooperativeLaunch << '\n'
        << "Cooperative Multi Device Launch: " << (bool)prop.cooperativeMultiDeviceLaunch << '\n'
        ;
}

int main()
{
    try
    {
        std::size_t device_count = pcw::device_count();
        if (device_count == 0)
        {
            std::cout << "No CUDA device(s) found.\n";
        }
        else
        {
            std::cout << device_count << " CUDA device(s) found.\n";
            for (std::size_t i = 0; i < pcw::device_count(); ++i)
            {
                int driverVersion = 0;
                int runtimeVersion = 0;

                print_device_info(i);
            }
        }
    }
    catch (std::exception const& ex)
    {
        std::cout << "exception: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
