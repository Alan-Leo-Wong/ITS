#pragma once

#if defined(__clang__) || defined(__GNUC__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#endif

#ifdef __GNUC__
#define ITS_FUNCTION __PRETTY_FUNCTION__
#elif defined(__clang__) || (_MSC_VER >= 1310)
#define ITS_FUNCTION __FUNCTION__
#else
#define GEOBOX_FUNCTION "unknown"
#endif

/* namespace macro */
#define ITS its

#if !defined(NAMESPACE_BEGIN)
#define NAMESPACE_BEGIN(name) namespace name {
#endif

#if !defined(NAMESPACE_END)
#define NAMESPACE_END(name) }
#endif

/* CUDA macro */
#ifdef __CUDACC__
#define CUDA_HOST_CALL __host__
#else
#define CUDA_HOST_CALL
#endif

#ifdef __CUDACC__
#define CUDA_DEVICE_CALL __device__
#else
#define CUDA_DEVICE_CALL
#endif

#ifdef __CUDACC__
#define CUDA_GENERAL_CALL __host__ __device__
#else
#define CUDA_GENERAL_CALL
#endif

#ifdef __CUDACC__
#define CUDA_KERNEL_CALL __global__
#else
#define CUDA_KERNEL_CALL
#endif
