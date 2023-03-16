#pragma once

inline double BaseFunction(const double& x, const double& nodePos, const double& nodeWidth)
{
	if (x <= nodePos - nodeWidth || x >= nodePos + nodeWidth) return 0.0;
	if (x <= nodePos) return 1 + (x - nodePos) / nodeWidth;
	if (x > nodePos) return 1 - (x - nodePos) / nodeWidth;
}
