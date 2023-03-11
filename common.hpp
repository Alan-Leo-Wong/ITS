#pragma once
#include <device_launch_parameters.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#define EMPTY_ARG   -1
#define INVALID_ARG -2

template <typename T>
CUDA_CALLABLE_MEMBER
bool isInRange(const double& l, const double& r, const T& query) {
	return l <= query && query <= r;
}

template <typename... T>
CUDA_CALLABLE_MEMBER
bool isInRange(const double& l, const double& r, const T &...query) {
	return isInRange(query...);
}