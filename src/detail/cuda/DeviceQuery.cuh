#pragma once

#include "CUDACheck.cuh"

NAMESPACE_BEGIN(ITS)

// Defines for GPU Architecture types (using the SM version to determine
// the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version,
        // and m = SM minor version
        int Cores;
    } sSMtoCores;

    static sSMtoCores nGpuArchCoresPerSM[] = {
            {0x30, 192},
            {0x32, 192},
            {0x35, 192},
            {0x37, 192},
            {0x50, 128},
            {0x52, 128},
            {0x53, 128},
            {0x60, 64},
            {0x61, 128},
            {0x62, 128},
            {0x70, 64},
            {0x72, 64},
            {0x75, 64},
            {0x80, 64},
            {0x86, 128},
            {0x87, 128},
            {-1,   -1}};

    inline int convertSMVer2Cores(int major, int minor) {
        int index = 0;
        while (nGpuArchCoresPerSM[index].SM != -1) {
            if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
                return nGpuArchCoresPerSM[index].Cores;
            index++;
        }
    }

    inline int getDeviceCount() {
        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            printf("There are no available device(s) that support CUDA\n");
            exit(EXIT_FAILURE);
        } else {
            printf("Detected %d CUDA Capable device(s)\n", deviceCount);
        }
        return deviceCount;
    }

    inline int getMaxComputeDevice() {
        int deviceCount = getDeviceCount();
        int maxNumSMs = 0, maxDevice = 0;
        if (deviceCount > 1) {
            for (int device = 0; device < deviceCount; ++device) {
                cudaDeviceProp prop;
                CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
                if (maxNumSMs < prop.multiProcessorCount) {
                    maxNumSMs = prop.multiProcessorCount;
                    maxDevice = device;
                }
            }
        }
        return maxDevice;
    }

    inline void getDeviceDefaultProp(int device) {
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

        printf("\nDevice %d: \"%s\"\n", device, deviceProp.name);

        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10,
               runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               deviceProp.major, deviceProp.minor);

        printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               convertSMVer2Cores(deviceProp.major, deviceProp.minor),
               convertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);

        printf("  Total amount of global memory:                 %.0f MBytes\n",
               (float) (deviceProp.totalGlobalMem / 1048576.0f));
        printf("  Total amount of constant memory:               %zu bytes\n",
               deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n",
               deviceProp.regsPerBlock);
    }

    inline void getDeviceDetailedProp(int device) {
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

        printf("\nDevice %d: \"%s\"\n", device, deviceProp.name);

        int driverVersion = 0, runtimeVersion = 0;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);

        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10,
               runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               deviceProp.major, deviceProp.minor);

        printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               convertSMVer2Cores(deviceProp.major, deviceProp.minor),
               convertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);

#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n",
               deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",
               deviceProp.memoryBusWidth);
        if (deviceProp.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n",
                   deviceProp.l2CacheSize);
        }
#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the
        // CUDA Driver API)
        int memoryClock;
        CUDA_CHECK(cuDeviceGetAttribute(
            &memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device));
        printf("  Memory Clock rate:                             %.0f Mhz\n",
            memoryClock * 1e-3f);
        int memBusWidth;
        CUDA_CHECK(cuDeviceGetAttribute(
            &memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device));
        printf("  Memory Bus Width:                              %d-bit\n",
            memBusWidth);
        int L2CacheSize;
        CUDA_CHECK(cuDeviceGetAttribute(&L2CacheSize,
            CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device));

        if (L2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n",
                L2CacheSize);
        }
#endif

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
               "%d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
               deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
               deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf(
                "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
                deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
               "layers\n",
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
               deviceProp.maxTexture2DLayered[2]);

        printf("  Total amount of global memory:                 %.0f MBytes\n",
               (float) (deviceProp.totalGlobalMem / 1048576.0f));
        printf("  Total amount of constant memory:               %zu bytes\n",
               deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %zu bytes\n",
               deviceProp.sharedMemPerBlock);
        printf("  Total shared memory per multiprocessor:        %zu bytes\n",
               deviceProp.sharedMemPerMultiprocessor);
        printf("  Total number of registers available per block: %d\n",
               deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n",
               deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %zu bytes\n",
               deviceProp.memPitch);
        printf("  Texture alignment:                             %zu bytes\n",
               deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy "
               "engine(s)\n",
               (deviceProp.deviceOverlap ? "Yes" : "No"),
               deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n",
               deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n",
               deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n",
               deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n",
               deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n",
               deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
               deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                    : "WDDM (Windows Display Driver Model)");
#endif
        printf("  Device supports Unified Addressing (UVA):      %s\n",
               deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Device supports Managed Memory:                %s\n",
               deviceProp.managedMemory ? "Yes" : "No");
        printf("  Device supports Compute Preemption:            %s\n",
               deviceProp.computePreemptionSupported ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n",
               deviceProp.cooperativeLaunch ? "Yes" : "No");
        printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
               deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
               deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char *sComputeMode[] = {
                "Default (multiple host threads can use ::cudaSetDevice() with device "
                "simultaneously)",
                "Exclusive (only one host thread in one process is able to use "
                "::cudaSetDevice() with this device)",
                "Prohibited (no host thread can use ::cudaSetDevice() with this "
                "device)",
                "Exclusive Process (many threads in one process is able to use "
                "::cudaSetDevice() with this device)",
                "Unknown",
                NULL};
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    inline void getDeviceProp(int device, int mode) {
        if (mode == 0)
            getDeviceDefaultProp(device);
        else
            getDeviceDetailedProp(device);
    }

NAMESPACE_END(ITS)
