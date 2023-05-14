#pragma once
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
	double x = BaseFunction(queryPoint.x(), nodePos.x(), nodeW);
	if (x <= 0.0) return 0.0;
	double y = BaseFunction(queryPoint.y(), nodePos.y(), nodeW);
	if (y <= 0.0) return 0.0;
	double z = BaseFunction(queryPoint.z(), nodePos.z(), nodeW);
	if (z <= 0.0) return 0.0;
	return x * y * z;
}
