#pragma once
#include <fstream>
#include <Eigen/dense>
#include "cuda/CUDAMacro.h"

// An Axis Aligned Box (AAB) of a certain Real - to be initialized with a boxOrigin and boxEnd
template <typename Real>
struct AABox {
	//using Real = typename Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;

	Real boxOrigin;
	Real boxEnd;
	Real boxWidth;

	_CUDA_GENERAL_CALL_ AABox() : boxOrigin(Real()), boxEnd(Real()), boxWidth(Real()) {}
	_CUDA_GENERAL_CALL_ AABox(const Real& _boxOrigin, const Real& _boxEnd) : boxOrigin(_boxOrigin), boxEnd(_boxEnd), boxWidth(_boxEnd - _boxOrigin) {}

	_CUDA_GENERAL_CALL_ void scaleAndTranslate(const double& scale_factor, const V3d& translation)
	{
		// 计算中心点
		const Real center = (boxOrigin + boxEnd) / 2.0;

		// 以中心点为基准进行缩放和平移
		const Real scaled_min_point = (boxOrigin - center) * scale_factor + center + translation;
		const Real scaled_max_point = (boxEnd - center) * scale_factor + center + translation;

		// 更新边界框的坐标
		boxOrigin = scaled_min_point;
		boxEnd = scaled_max_point;
	}

	_CUDA_GENERAL_CALL_ AABox<Real>(const AABox<Real>& _box)
	{
		boxOrigin = _box.boxOrigin;
		boxEnd = _box.boxEnd;
		boxWidth = _box.boxWidth;
	}

	_CUDA_GENERAL_CALL_ AABox<Real>& operator=(const AABox<Real>& _box)
	{
		boxOrigin = _box.boxOrigin;
		boxEnd = _box.boxEnd;
		boxWidth = _box.boxWidth;

		return *this;
	}
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
