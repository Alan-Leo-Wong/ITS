#pragma once
/*
* 一些公用 CUDA 计算
*/
#include "..\BasicDataType.h"
#include "..\utils\Geometry.hpp"
#include "..\utils\cuda\CUDACheck.cuh"
#include "..\utils\cuda\cuBLASCheck.cuh"
#include <thrust\device_vector.h>

namespace cuAcc {
	//void accIntersection();

	void launch_modelTriAttributeKernel(const size_t& nTriangles,
		std::vector<Triangle<Eigen::Vector3d>>& modelTriangleArray);

	//template <typename T>
	void launch_BLASRowSumReduce(const int& rows,
		const int& columns,
		double* d_matrix,
		double* d_row_sums,
		const cudaStream_t& stream = nullptr);

	//template <typename T>
	void launch_ThrustRowSumReduce(const int& rows,
		const int& columns,
		const thrust::device_vector<double>& d_matrix,
		thrust::device_vector<double>& row_sums,
		const cudaStream_t& stream = nullptr);

	void cpBSplineVal(const uint& numPoints, const uint& numNodeVerts, const uint& numNodes,
		const std::vector<V3d>& pointsData, const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>>& nodeVertexArray,
		const std::vector<V3d>& nodeWidthArray, const VXd& lambda, VXd& bSplinVal, const bool& useThrust = true);

	void cpBSplineVal(const cudaDeviceProp& prop, const uint& numPoints, const uint& numNodeVerts,
		const thrust::pair<Eigen::Vector3d, uint32_t>* d_nodeVertexArray, const V3d* d_nodeWidthArray,
		const double* d_lambda, const thrust::device_vector<V3d>& d_pointsData,
		thrust::device_vector<double>& d_bSplineVal, const cudaStream_t& stream, const bool& useThrust = true);
}
