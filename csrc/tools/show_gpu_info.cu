#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void check(cudaError_t err, const char* call) {
    if (err != cudaSuccess) {
        std::cerr << call << " failed: " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

std::string format_bytes(unsigned long long bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;

    while (value >= 1024.0 && unit < 4) {
        value /= 1024.0;
        ++unit;
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value << " " << units[unit];
    return out.str();
}

void print_row(const std::string& key, const std::string& value) {
    std::cout << std::left << std::setw(42) << key << "  " << value << "\n";
}

void print_row(const std::string& key, int value) {
    print_row(key, std::to_string(value));
}

void print_attr(int device, const std::string& name, cudaDeviceAttr attr) {
    int value = 0;
    cudaError_t err = cudaDeviceGetAttribute(&value, attr, device);

    if (err == cudaSuccess) {
        print_row(name, value);
        return;
    }

    // Clear the thread-local CUDA error slot so one unsupported attribute does
    // not affect later runtime calls.
    cudaGetLastError();
    print_row(name, "n/a (" + std::string(cudaGetErrorString(err)) + ")");
}

void print_device(int device) {
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    std::cout << "Device " << device << "\n";
    std::cout << "--------\n";
    print_row("name", prop.name);
    print_row("compute capability", std::to_string(prop.major) + "." + std::to_string(prop.minor));
    print_row("total global memory", format_bytes(prop.totalGlobalMem));
    print_row("shared memory / block", format_bytes(prop.sharedMemPerBlock));
    print_row("registers / block", prop.regsPerBlock);
    print_row("warp size", prop.warpSize);
    print_row("max threads / block", prop.maxThreadsPerBlock);
    print_row("max block dim", std::to_string(prop.maxThreadsDim[0]) + " x " +
                                   std::to_string(prop.maxThreadsDim[1]) + " x " +
                                   std::to_string(prop.maxThreadsDim[2]));
    print_row("max grid dim", std::to_string(prop.maxGridSize[0]) + " x " +
                                  std::to_string(prop.maxGridSize[1]) + " x " +
                                  std::to_string(prop.maxGridSize[2]));
    print_row("clock rate", std::to_string(prop.clockRate) + " kHz");
    print_row("memory clock rate", std::to_string(prop.memoryClockRate) + " kHz");
    print_row("memory bus width", std::to_string(prop.memoryBusWidth) + " bits");
    print_row("L2 cache", format_bytes(prop.l2CacheSize));
    print_row("SM count", prop.multiProcessorCount);
    print_row("concurrent kernels", prop.concurrentKernels);
    print_row("async engine count", prop.asyncEngineCount);
    print_row("unified addressing", prop.unifiedAddressing);
    print_row("ECC enabled", prop.ECCEnabled);

    std::cout << "\nCUDA Runtime Attributes\n";
    std::cout << "-----------------------\n";
    print_attr(device, "compute capability major", cudaDevAttrComputeCapabilityMajor);
    print_attr(device, "compute capability minor", cudaDevAttrComputeCapabilityMinor);
    print_attr(device, "multiprocessor count", cudaDevAttrMultiProcessorCount);
    print_attr(device, "warp size", cudaDevAttrWarpSize);
    print_attr(device, "max threads / block", cudaDevAttrMaxThreadsPerBlock);
    print_attr(device, "max threads / multiprocessor", cudaDevAttrMaxThreadsPerMultiProcessor);
    print_attr(device, "max blocks / multiprocessor", cudaDevAttrMaxBlocksPerMultiprocessor);
    print_attr(device, "registers / block", cudaDevAttrMaxRegistersPerBlock);
    print_attr(device, "registers / multiprocessor", cudaDevAttrMaxRegistersPerMultiprocessor);
    print_attr(device, "shared memory / block", cudaDevAttrMaxSharedMemoryPerBlock);
    print_attr(device, "shared memory / block opt-in", cudaDevAttrMaxSharedMemoryPerBlockOptin);
    print_attr(device, "shared memory / multiprocessor", cudaDevAttrMaxSharedMemoryPerMultiprocessor);
    print_attr(device, "reserved shared memory / block", cudaDevAttrReservedSharedMemoryPerBlock);
    print_attr(device, "L2 cache size", cudaDevAttrL2CacheSize);
    print_attr(device, "memory clock rate", cudaDevAttrMemoryClockRate);
    print_attr(device, "global memory bus width", cudaDevAttrGlobalMemoryBusWidth);
    print_attr(device, "max pitch", cudaDevAttrMaxPitch);
    print_attr(device, "texture alignment", cudaDevAttrTextureAlignment);
    print_attr(device, "concurrent kernels", cudaDevAttrConcurrentKernels);
    print_attr(device, "async engine count", cudaDevAttrAsyncEngineCount);
    print_attr(device, "unified addressing", cudaDevAttrUnifiedAddressing);
    print_attr(device, "managed memory", cudaDevAttrManagedMemory);
    print_attr(device, "pageable memory access", cudaDevAttrPageableMemoryAccess);
    print_attr(device, "concurrent managed access", cudaDevAttrConcurrentManagedAccess);
    print_attr(device, "cooperative launch", cudaDevAttrCooperativeLaunch);
    print_attr(device, "cooperative multi-device launch", cudaDevAttrCooperativeMultiDeviceLaunch);
    print_attr(device, "compute preemption supported", cudaDevAttrComputePreemptionSupported);
    print_attr(device, "can use host pointer for registered mem", cudaDevAttrCanUseHostPointerForRegisteredMem);
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    if (argc > 2) {
        std::cerr << "usage: " << argv[0] << " [device]\n";
        return 2;
    }
    if (argc == 2) {
        device = std::atoi(argv[1]);
    }

    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }
    if (device < 0 || device >= count) {
        std::cerr << "Invalid device " << device << "; available devices: 0.." << (count - 1) << "\n";
        return 2;
    }

    check(cudaSetDevice(device), "cudaSetDevice");
    print_device(device);
    return 0;
}
