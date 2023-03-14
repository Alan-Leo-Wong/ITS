#pragma once
#include "../cuAcc/CUDAMacro.h"

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