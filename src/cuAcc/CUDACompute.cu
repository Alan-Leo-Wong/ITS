#include "CUDACompute.h"
#include "..\BSpline.hpp"
#include "..\utils\cuda\CUDAUtil.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <thrust\reduce.h>
#include <thrust\generate.h>
#include <thrust\device_vector.h>
#include <device_launch_parameters.h>

//__global__ void intersectionKernel(const MXd d_V,
//	const V2i* d_modelEdges,
//	const OctreeNode** d_leafNodes,
//	const int edgesNum,
//	const int leafNodesNum,
//	V3d* d_intersections)
//{
//	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
//	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (tx < edgesNum && ty < leafNodesNum)
//	{
//		Eigen::Vector2i e = d_modelEdges[tx];
//		V3d p1 = d_V.row(e.x());
//		V3d dir = d_V.row(e.y()) - p1;
//
//		auto node = d_leafNodes[ty];
//
//		V3d leftBottomBackCorner = node->boundary.first;
//
//		double DINF = cuda::std::numeric_limits<double>::max();
//
//		// bottom plane
//		double bottom_t = DINF;
//		if (dir.z() != 0)
//			bottom_t = (leftBottomBackCorner.z() - p1.z()) / dir.z();
//		// left plane
//		double left_t = DINF;
//		if (dir.y() != 0)
//			left_t = (leftBottomBackCorner.y() - p1.y()) / dir.y();
//		// back plane
//		double back_t = DINF;
//		if (dir.x() != 0)
//			back_t = (leftBottomBackCorner.x() - p1.x()) / dir.x();
//
//		int id = 0;
//		if (isInRange(.0, 1.0, bottom_t))
//		{
//			d_intersections[tx * 3 + id] = p1 + bottom_t * dir;
//			++id;
//		}
//		if (isInRange(.0, 1.0, left_t))
//		{
//			d_intersections[tx * 3 + id] = p1 + left_t * dir;
//			++id;
//		}
//		if (isInRange(.0, 1.0, back_t))
//		{
//			d_intersections[tx * 3 + id] = p1 + back_t * dir;
//			++id;
//		}
//	}
//}
//
//void accIntersection()
//{
//
//}

namespace cuAcc {
	// 计算三角形法线和面积
	template<typename Real>
	__global__ void modelTriAttributeKernel(const size_t nTriangles,
		Triangle<Real>* d_modelTriangleArray)
	{
		const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < nTriangles)
		{
			const Real p1 = d_modelTriangleArray[tid].p1;
			const Real p2 = d_modelTriangleArray[tid].p2;
			const Real p3 = d_modelTriangleArray[tid].p3;

			const Real normal = (p2 - p1).cross(p3 - p1);

			d_modelTriangleArray[tid].normal = normal;
			d_modelTriangleArray[tid].area = 0.5 * (normal.norm());
			d_modelTriangleArray[tid].dir = (-normal).dot(p1);
		}
	}

	//template<typename Real>

	void launch_modelTriAttributeKernel(const size_t& nTriangles,
		std::vector<Triangle<Eigen::Vector3d>>& modelTriangleArray)
	{
		thrust::device_vector<Triangle<Eigen::Vector3d>> d_modelTriangleArray = modelTriangleArray;
		int blockSize, gridSize, minGridSize;
		getOccupancyMaxPotentialBlockSize(nTriangles, minGridSize, blockSize, gridSize, modelTriAttributeKernel<Eigen::Vector3d>, 0, 0);
		modelTriAttributeKernel << <gridSize, blockSize >> > (nTriangles, d_modelTriangleArray.data().get());
		getLastCudaError("Kernel 'modelTriAttributeKernel' launch failed!\n");

		CUDA_CHECK(cudaMemcpy(modelTriangleArray.data(), d_modelTriangleArray.data().get(), sizeof(Triangle<Eigen::Vector3d>) * nTriangles, cudaMemcpyDeviceToHost));
	}

	void launch_BLASRowSumReduce(const int& rows,
		const int& columns,
		double* d_matrix,
		double* d_row_sums,
		const cudaStream_t& stream)
	{
		cublasOperation_t transa = CUBLAS_OP_N;
		cublasOperation_t transb = CUBLAS_OP_N;

		const double alpha = 1.0;
		const double beta = 0.0;

		const int lda = 1;
		const int ldb = columns;
		const int ldc = lda;

		cublasHandle_t cublasH = nullptr;
		CUBLAS_CHECK(cublasCreate(&cublasH));

		if (stream != nullptr)
			CUBLAS_CHECK(cublasSetStream(cublasH, stream));

		std::vector<double> identityVector(columns, (double)1.0); // 全1的向量
		double* d_vec;
		CUDA_CHECK(cudaMalloc((void**)&d_vec, sizeof(double) * columns));
		CUDA_CHECK(cudaMemcpyAsync(d_vec, identityVector.data(), sizeof(double) * columns, cudaMemcpyHostToDevice, stream));

		CUBLAS_CHECK(cublasDgemm(cublasH, transa, transb, 1, rows, columns, &alpha, d_vec, lda, d_matrix, ldb, &beta, d_row_sums, ldc));

		CUBLAS_CHECK(cublasDestroy(cublasH));
		CUDA_CHECK(cudaFree(d_vec));
	}

	// convert a linear index to a row index
	template <typename T>
	struct linear_index_to_row_index : public thrust::unary_function<T, T>
	{
		T C; // number of columns

		__host__ __device__
			linear_index_to_row_index(T C) : C(C) {}

		__host__ __device__
			T operator()(T i)
		{
			return i / C;
		}
	};

	void launch_ThrustRowSumReduce(const int& rows,
		const int& columns,
		const thrust::device_vector<double>& d_matrix,
		thrust::device_vector<double>& row_sums,
		const cudaStream_t& stream) // thrust::universal_vector
	{
		assert(row_sums.size() == rows, "THE SIZE OF ROW_SUMS IS NOT EQUAL ROWS!\n");

		thrust::device_vector<int> row_indices(rows);

		if (stream)
			thrust::reduce_by_key(thrust::cuda::par.on(stream),
				thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(columns)),
				thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(rows)) + (rows * columns),
				d_matrix.begin(),
				row_indices.begin(),
				row_sums.begin(),
				thrust::equal_to<int>(),
				thrust::plus<double>());
		else
			thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(columns)),
				thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(columns)) + (rows * columns),
				d_matrix.begin(),
				row_indices.begin(),
				row_sums.begin(),
				thrust::equal_to<int>(),
				thrust::plus<double>());
	}

	__global__ void prepareMatrixKernel(const uint dataSize, const uint numNodeVerts,
		const Eigen::Vector3d* d_pointsData, const thrust::pair<Eigen::Vector3d, uint32_t>* d_nodeVertexArray,
		const V3d* d_nodeWidthArray, const double* d_lambda, double* d_outMatrix)
	{
		const size_t tx = threadIdx.x + blockIdx.x * blockDim.x;
		const size_t ty = threadIdx.y + blockIdx.y * blockDim.y;

		if (tx < numNodeVerts && ty < dataSize)
		{
			const size_t idx = ty * numNodeVerts + tx;

			d_outMatrix[idx] = d_lambda[tx] * BaseFunction4Point(d_nodeVertexArray[tx].first,
				d_nodeWidthArray[d_nodeVertexArray[tx].second], d_pointsData[ty]);
		}
	}

	// argument is in host
	void cpBSplineVal(const uint& numPoints, const uint& numNodeVerts, const uint& numNodes,
		const std::vector<V3d>& pointsData, const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>>& nodeVertexArray,
		const std::vector<V3d>& nodeWidthArray, const VXd& lambda, VXd& bSplinVal, const bool& useThrust)
	{
		dim3 blockSize(16, 64, 1);
		dim3 gridSize((numNodeVerts + blockSize.x - 1) / blockSize.x, (numPoints + blockSize.y - 1) / blockSize.y, 1);

		V3d* d_pointsData;
		thrust::pair<Eigen::Vector3d, uint32_t>* d_nodeVertexArray;
		V3d* d_nodeWidthArray;
		double* d_lambda;
		thrust::device_vector<double> d_matrix(numPoints * numNodeVerts);

		CUDA_CHECK(cudaMalloc((void**)&d_pointsData, sizeof(V3d) * numPoints));
		CUDA_CHECK(cudaMalloc((void**)&d_nodeVertexArray, sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts));
		CUDA_CHECK(cudaMalloc((void**)&d_nodeWidthArray, sizeof(V3d) * numNodes));
		CUDA_CHECK(cudaMalloc((void**)&d_lambda, sizeof(double) * numNodeVerts));

		CUDA_CHECK(cudaMemcpy(d_pointsData, pointsData.data(), sizeof(V3d) * numPoints, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_nodeVertexArray, nodeVertexArray.data(), sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_nodeWidthArray, nodeWidthArray.data(), sizeof(V3d) * numNodes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_lambda, lambda.data(), sizeof(double) * numNodeVerts, cudaMemcpyHostToDevice));

		prepareMatrixKernel << <gridSize, blockSize >> > (numPoints, numNodeVerts, d_pointsData, d_nodeVertexArray, d_nodeWidthArray, d_lambda, d_matrix.data().get());
		getLastCudaError("Kernel 'prepareMatrixKernel' launch failed!\n");

		CUDA_CHECK(cudaFree(d_pointsData));
		CUDA_CHECK(cudaFree(d_nodeVertexArray));
		CUDA_CHECK(cudaFree(d_nodeWidthArray));
		CUDA_CHECK(cudaFree(d_lambda));

		double* h_bSplineVal = new double(numPoints);
		if (useThrust)
		{
			thrust::device_vector<double> d_bSplineVal(numPoints);
			launch_ThrustRowSumReduce(numPoints, numNodeVerts, d_matrix, d_bSplineVal, nullptr);
			CUDA_CHECK(cudaMemcpy(h_bSplineVal, d_bSplineVal.data().get(), sizeof(double) * numPoints, cudaMemcpyDeviceToHost));
		}
		else
		{
			double* d_bSplineVal;
			CUDA_CHECK(cudaMalloc((void**)&d_bSplineVal, sizeof(V3d) * numPoints));
			launch_BLASRowSumReduce(numPoints, numNodeVerts, d_matrix.data().get(), d_bSplineVal, nullptr);

			CUDA_CHECK(cudaMemcpy(h_bSplineVal, d_bSplineVal, sizeof(double) * numPoints, cudaMemcpyDeviceToHost));
			CUDA_CHECK(cudaFree(d_bSplineVal));
		}
		cleanupThrust(d_matrix);

		bSplinVal = Eigen::Map<Eigen::VectorXd>(h_bSplineVal, numPoints);
		free(h_bSplineVal);
	}

	// argument is in device
	void cpBSplineVal(const uint& numPoints, const uint& numNodeVerts, 
		const thrust::pair<Eigen::Vector3d, uint32_t>* d_nodeVertexArray, const V3d* d_nodeWidthArray, 
		const double* d_lambda, thrust::device_vector<V3d>& d_pointsData, 
		thrust::device_vector<double>& d_bSplineVal, const bool& useThrust)
	{
		dim3 blockSize(16, 64, 1);
		dim3 gridSize((numNodeVerts + blockSize.x - 1) / blockSize.x, (numPoints + blockSize.y - 1) / blockSize.y, 1);

		thrust::device_vector<double> d_matrix(numPoints * numNodeVerts);

		prepareMatrixKernel << <gridSize, blockSize >> > (numPoints, numNodeVerts, d_pointsData.data().get(),
			d_nodeVertexArray, d_nodeWidthArray, d_lambda, d_matrix.data().get());
		getLastCudaError("Kernel 'prepareMatrixKernel' launch failed!\n");

		cleanupThrust(d_pointsData); // avoid out of memory

		if (d_bSplineVal.size() != numPoints) { d_bSplineVal.clear(); resizeThrust(d_bSplineVal, numPoints); }
		if (useThrust)
		{
			launch_ThrustRowSumReduce(numPoints, numNodeVerts, d_matrix, d_bSplineVal, nullptr);
		}
		else
		{
			launch_BLASRowSumReduce(numPoints, numNodeVerts, d_matrix.data().get(), d_bSplineVal.data().get(), nullptr);
		}

		cleanupThrust(d_matrix);
	}
}