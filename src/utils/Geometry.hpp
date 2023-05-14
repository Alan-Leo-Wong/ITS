#pragma once
#include <fstream>
#include <Eigen/dense>
#include "cuda/CUDAMacro.h"

// An Axis Aligned Box (AAB) of a certain type - to be initialized with a boxOrigin and boxEnd
template <typename Real>
struct AABox {
	Real boxOrigin;
	Real boxEnd;
	Real boxWidth;

	_CUDA_GENERAL_CALL_ AABox() : boxOrigin(Real()), boxEnd(Real()), boxWidth(Real()) {}
	_CUDA_GENERAL_CALL_ AABox(Real _boxOrigin, Real _boxEnd) : boxOrigin(_boxOrigin), boxEnd(_boxEnd), boxWidth(_boxEnd - _boxOrigin) {}
};

template <typename Real>
struct Triangle
{
	Real p1, p2, p3;
	Real normal;
	double area;
	double dir;

	_CUDA_GENERAL_CALL_ Triangle() {}

	_CUDA_GENERAL_CALL_ Triangle(const Real& _p1, const Real& _p2, const Real& _p3) :p1(_p1), p2(_p2), p3(_p3) {}
};
