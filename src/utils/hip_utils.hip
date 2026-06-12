#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>

// HIP error checking macro
#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", \
                __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// GPU device info
struct GpuInfo {
    char name[256];
    int compute_units;
    size_t global_mem;
    size_t shared_mem_per_block;
    int warp_size;
    int max_threads_per_block;
};

GpuInfo get_gpu_info() {
    GpuInfo info;
    hipDevice_t device;
    hipDeviceGet(&device, 0);
    hipDeviceGetName(info.name, 256, device);
    hipDeviceAttribute_t attr;
    hipDeviceGetAttribute(&attr, hipDeviceAttributeMultiprocessorCount, device);
    info.compute_units = attr;
    hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxSharedMemoryPerBlock, device);
    info.shared_mem_per_block = attr;
    hipDeviceGetAttribute(&attr, hipDeviceAttributeWarpSize, device);
    info.warp_size = attr;
    hipDeviceGetAttribute(&attr, hipDeviceAttributeMaxThreadsPerBlock, device);
    info.max_threads_per_block = attr;
    size_t free, total;
    hipMemGetInfo(&free, &total);
    info.global_mem = total;
    return info;
}

void print_gpu_info() {
    GpuInfo info = get_gpu_info();
    printf("=== GPU Info ===\n");
    printf("  Device:       %s\n", info.name);
    printf("  Compute Units:%d\n", info.compute_units);
    printf("  Global Memory:%.1f GB\n", info.global_mem / (1024.0*1024.0*1024.0));
    printf("  Shared Memory:%zu KB\n", info.shared_mem_per_block / 1024);
    printf("  Warp Size:    %d\n", info.warp_size);
    printf("  Max Threads:  %d\n", info.max_threads_per_block);
    printf("================\n");
}

// GPU timer using hip events
struct GpuTimer {
    hipEvent_t start, stop;
    
    GpuTimer() {
        hipEventCreate(&start);
        hipEventCreate(&stop);
    }
    ~GpuTimer() {
        hipEventDestroy(start);
        hipEventDestroy(stop);
    }
    void begin() { hipEventRecord(start); }
    float end() {
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float ms;
        hipEventElapsedTime(&ms, start, stop);
        return ms;
    }
};
