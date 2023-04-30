#include <vector>
#include <cuda_runtime.h>
#include <thrust\device_vector.h>
#include <device_launch_parameters.h>
#include "CUDACompute.h"
#include "..\BasicDataType.h"
#include "..\utils\Geometry.hpp"

void launch_BLASRowSumReduce(const cudaStream_t& stream,
	const int& rows,
	const int& columns,
	double* d_matrix,
	double* d_res)
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

	CUBLAS_CHECK(cublasDgemm(cublasH, transa, transb, 1, rows, columns, &alpha, d_vec, lda, d_matrix, ldb, &beta, d_res, ldc));

	CUBLAS_CHECK(cublasDestroy(cublasH));
	CUDA_CHECK(cudaFree(d_vec));
}

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

// 计算三角形法线和面积
template<typename Real>
__global__ void modelTriAttributeKernel(const size_t& nTriangles,
	Triangle<Real>* d_modelTriangleArray)
{
	const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < nTriangles)
	{
		const V3d p1 = d_modelTriangleArray[tid].p1;
		const V3d p2 = d_modelTriangleArray[tid].p2;
		const V3d p3 = d_modelTriangleArray[tid].p3;

		const V3d normal = (p2 - p1).cross(p3 - p1);

		d_modelTriangleArray[tid].normal = normal;
		d_modelTriangleArray[tid].area = 0.5 * (normal.norm());
		d_modelTriangleArray[tid].dir = (-normal).dot(p1);
	}
}

template<typename Real>
void launch_modelTriAttributeKernel(const size_t& nTriangles,
	std::vector<Triangle<Real>>& modelTriangleArray)
{
	thrust::device_vector<Triangle<Real>>d_modelTriangleArray(modelTriangleArray);
	int blockSize, gridSize;

	modelTriAttributeKernel << <gridSize, blockSize >> > (nTriangles, d_modelTriangleArray.data().get());

	CUDA_CHECK(cudaMemcpy(modelTriangleArray.data(), d_modelTriangleArray.data().get(), sizeof(Triangle<Real>) * nTriangles, cudaMemcpyDeviceToHost));
}