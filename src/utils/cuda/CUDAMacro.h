#include <device_launch_parameters.h>

#ifdef __CUDACC__
#define _CUDA_HOST_CALL_ __host__
#else
#define _CUDA_HOST_CALL_
#endif

#ifdef __CUDACC__
#define _CUDA_DEVICE_CALL_ __device__
#else
#define _CUDA_DEVICE_CALL_
#endif

#ifdef __CUDACC__
#define _CUDA_GENERAL_CALL_ __host__ __device__
#else
#define _CUDA_GENERAL_CALL_
#endif

#ifdef __CUDACC__
#define _CUDA_KERNEL_CALL_ __global__
#else
#define _CUDA_KERNEL_CALL_
#endif