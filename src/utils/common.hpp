#pragma once
#include "../cuAcc/CUDAMacro.h"

template <typename T>
CUDA_CALLABLE_MEMBER
inline bool isInRange(const double& l, const double& r, const T& query) {
	return l <= query && query <= r;
}

template <typename... T>
CUDA_CALLABLE_MEMBER
inline bool isInRange(const double& l, const double& r, const T &...query) {
	return isInRange(query...);
}

template <typename Scalar, typename Derived>
CUDA_CALLABLE_MEMBER 
inline bool list2Matrix(const std::vector<Scalar>& V, Eigen::PlainObjectBase<Derived>& M)
{
    // number of rows
    int m = V.size();
    if (m == 0)
    {
        //fprintf(stderr,"Error: list_to_matrix() list is empty()\n");
        //return false;
        if (Derived::ColsAtCompileTime == 1) M.resize(0, 1);
        else if (Derived::RowsAtCompileTime == 1) M.resize(1, 0);
        else M.resize(0, 0);

        return true;
    }
    // Resize output
    if (Derived::RowsAtCompileTime == 1) M.resize(1, m);
    else M.resize(m, 1);

    // Loop over rows
    for (int i = 0; i < m; i++)
        M(i) = V[i];

    return true;
}