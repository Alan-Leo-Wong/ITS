#pragma once
#include "BasicDataType.h"
#include "utils/cuda/CUDAMacro.h"

inline _CUDA_GENERAL_CALL_ double BaseFunction(const double& x, const double& nodePos, const double& nodeWidth)
{
	if (fabs(x - (nodePos - nodeWidth)) < 1e-9 || x < nodePos - nodeWidth ||
		fabs(x - (nodePos + nodeWidth)) < 1e-9 || x > nodePos + nodeWidth) return 0.0;
	if (x <= nodePos) return 1 + (x - nodePos) / nodeWidth;
	if (x > nodePos) return 1 - (x - nodePos) / nodeWidth;
}

inline _CUDA_GENERAL_CALL_ double BaseFunction4Point(const V3d& nodePos, const V3d& nodeW, const V3d& queryPoint)
{
	double x = BaseFunction(queryPoint.x(), nodePos.x(), nodeW.x());
	if (x <= 0.0) return 0.0;
	double y = BaseFunction(queryPoint.y(), nodePos.y(), nodeW.y());
	if (y <= 0.0) return 0.0;
	double z = BaseFunction(queryPoint.z(), nodePos.z(), nodeW.z());
	if (z <= 0.0) return 0.0;
	return x * y * z;
}

inline _CUDA_GENERAL_CALL_ double BaseFunction4Point(const V3d& nodePos, const double& nodeW, const V3d& queryPoint)
{
	return BaseFunction4Point(nodePos, V3d(nodeW, nodeW, nodeW), queryPoint);
}

inline _CUDA_GENERAL_CALL_ double de_BaseFunction(const double& x, const double& nodePos, const double& nodeWidth)
{
	if (fabs(x - (nodePos - nodeWidth)) < 1e-9 || x < nodePos - nodeWidth ||
		fabs(x - (nodePos + nodeWidth)) < 1e-9 || x > nodePos + nodeWidth) return 0.0;
	if (x <= nodePos) return 1 / nodeWidth;
	if (x > nodePos) return -1 / nodeWidth;
}

inline _CUDA_GENERAL_CALL_ V3d de_BaseFunction4Point(const V3d& nodePos, const V3d& nodeW, const V3d& queryPoint)
{
	V3d gradient;

	gradient.x() = de_BaseFunction(queryPoint.x(), nodePos.x(), nodeW.x())
		* BaseFunction(queryPoint.y(), nodePos.y(), nodeW.y())
		* BaseFunction(queryPoint.z(), nodePos.z(), nodeW.z());

	gradient.y() = BaseFunction(queryPoint.x(), nodePos.x(), nodeW.x())
		* de_BaseFunction(queryPoint.y(), nodePos.y(), nodeW.y())
		* BaseFunction(queryPoint.z(), nodePos.z(), nodeW.z());

	gradient.z() = BaseFunction(queryPoint.x(), nodePos.x(), nodeW.x())
		* BaseFunction(queryPoint.y(), nodePos.y(), nodeW.y())
		* de_BaseFunction(queryPoint.z(), nodePos.z(), nodeW.z());

	return gradient;
}

inline _CUDA_GENERAL_CALL_ V3d de_BaseFunction4Point(const V3d& nodePos, const double& nodeW, const V3d& queryPoint)
{
	return de_BaseFunction4Point(nodePos, V3d(nodeW, nodeW, nodeW), queryPoint);
}

inline _CUDA_GENERAL_CALL_ MXd de_BaseFunction4PointMat(const MXd& nodePos, const V3d& nodeW, const MXd& queryPointMat)
{
	MXd gradientMat(queryPointMat.rows(), 3);
	for (int i = 0; i < queryPointMat.rows(); ++i)
		gradientMat.row(i) = de_BaseFunction4Point(nodePos.row(i), nodeW(i), queryPointMat);
	return gradientMat;
}
