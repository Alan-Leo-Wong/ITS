#include "CUDACompute.h"
#include "..\BSpline.hpp"
#include "..\utils\cuda\CUDAUtil.cuh"
#include "..\utils\cuda\DeviceQuery.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <thrust\reduce.h>
#include <thrust\generate.h>
#include <thrust\device_vector.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

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
namespace cg = cooperative_groups;

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
		modelTriAttributeKernel<Eigen::Vector3d> << <gridSize, blockSize >> > (nTriangles, d_modelTriangleArray.data().get());
		getLastCudaError("Kernel 'modelTriAttributeKernel' launch failed!\n");

		CUDA_CHECK(cudaMemcpy(modelTriangleArray.data(), d_modelTriangleArray.data().get(), sizeof(Triangle<Eigen::Vector3d>) * nTriangles, cudaMemcpyDeviceToHost));
	}

	//template <typename T>
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

	//template <typename T>
	void launch_ThrustRowSumReduce(const int& rows,
		const int& columns,
		const thrust::device_vector<double>& d_matrix,
		thrust::device_vector<double>& row_sums,
		const cudaStream_t& stream) // thrust::universal_vector
	{
		if (row_sums.size() != rows) { row_sums.clear(); row_sums.resize(rows); }
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

	// sum reduce at warp level
	template <typename T>
	__device__ __forceinline__ void warpReduceSum(unsigned int mask, T& sum)
	{
		for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
			sum += __shfl_down_sync(mask, sum, offset);
	}

	/*
	 * matrix reduce for sum of row
	 * @param m: rows
	 * @param n: columns, and n % warpSize = 0
	 * @param g_idata: matrix(m * n)
	 */
	template <typename T = Eigen::Vector3d, typename Scalar = double, bool nIsPow2, unsigned int colBlockSize>
	__global__ void reduceRowSumKernel(const unsigned int m, const unsigned int n,
		const thrust::pair<T, uint32_t>* __restrict__ d_nodeVertexArray,
		const T* __restrict__ d_nodeWidthArray,
		const T* __restrict__ g_iA,
		const Scalar* __restrict__ g_iB, // lambda
		Scalar* __restrict__ g_odata)
	{
		Scalar* shData = SharedMemory<Scalar>();
		cg::thread_block ctb = cg::this_thread_block();

		unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
		unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;

		if (ty < m) {
			unsigned int x_tid = threadIdx.x;
			unsigned int x_gridSize = colBlockSize * gridDim.x;

			unsigned int maskLength = (colBlockSize & 31);
			maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
			const unsigned int mask = (0xffffffff) >> maskLength;

			Scalar sum = (Scalar).0;

			//printf("#1 sum = %lf\n", sum);
			// reduce multiple elements per thread
			if (nIsPow2) {
				unsigned int i = blockIdx.x * colBlockSize * 2 + threadIdx.x;
				x_gridSize <<= 1;

				while (i < n)
				{
					//printf("#2 sum = %lf\n", sum);
					sum += g_iB[i] * BaseFunction4Point(d_nodeVertexArray[i].first, d_nodeWidthArray[d_nodeVertexArray[i].second], g_iA[ty]);
					if (i + colBlockSize < n)
					{
						sum += g_iB[i] * BaseFunction4Point(d_nodeVertexArray[i].first, d_nodeWidthArray[d_nodeVertexArray[i].second], g_iA[ty]); // (一个)线程块级别的跨度
						i += x_gridSize; // 网格级别的跨度：默认网格大小(block的数量)为原有数据(x维度即列数)的一半(如果nIsPow2成立，则x_gridSize扩大一倍)
					}

				}
			}
			else {
				unsigned int i = blockIdx.x * colBlockSize + threadIdx.x;
				while (i < n)
				{
					//printf("#2 sum = %lf\n", sum);

					sum += g_iB[i] * BaseFunction4Point(d_nodeVertexArray[i].first, d_nodeWidthArray[d_nodeVertexArray[i].second], g_iA[ty]);
					i += x_gridSize;
				}
			}

			// 对每个warp执行归约求和，然后保存到shared memory中
			warpReduceSum<Scalar>(mask, sum);
			const int sh_reduceNum = (colBlockSize / warpSize) > 0 ? colBlockSize / warpSize : 1;
			if (x_tid % warpSize == 0)
				shData[threadIdx.y * sh_reduceNum + x_tid / warpSize] = sum;

			cg::sync(ctb);

			//printf("#3 sum = %lf\n", sum);

			// 同一个block下所有warp求和(只要将每个warp的第一个thread保存的sum加起来即可，
			// 因为每个warp的第一个thread保存的sum就是其所属warp的所有线程的数据和)
			const unsigned int newMask = __ballot_sync(mask, x_tid < sh_reduceNum);
			if (x_tid < sh_reduceNum) {
				sum = shData[threadIdx.y * sh_reduceNum + x_tid];
				warpReduceSum<Scalar>(newMask, sum);

				//printf("#4 sum = %lf\n", sum);

			}

			if (x_tid == 0) {
				g_odata[ty * gridDim.x + blockIdx.x] = sum;

				//printf("#4 ty = %d, sum = %lf\n", ty, sum);
			}
		}
	}

	template <typename T = Eigen::Vector3d, typename Scalar = double>
	void switchKernel(const bool& isPow2, const int& threads, const dim3& gridSize,
		const dim3& blockSize, const int& sh_memSize, const cudaStream_t& stream, const int& rowElems,
		const uint& cols, const thrust::pair<T, uint32_t>* d_nodeVertexArray,
		const T* d_nodeWidthArray, const T* d_A, const Scalar* d_B, Scalar* d_tRowSumMatrix)
	{
		if (isPow2) {
			switch (threads) {
			case 1024:
				reduceRowSumKernel<T, Scalar, true, 1024>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 512:
				reduceRowSumKernel<T, Scalar, true, 512>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 256:
				reduceRowSumKernel<T, Scalar, true, 256>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 128:
				reduceRowSumKernel<T, Scalar, true, 128>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 64:
				reduceRowSumKernel<T, Scalar, true, 64>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 32:
				reduceRowSumKernel<T, Scalar, true, 32>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 16:
				reduceRowSumKernel<T, Scalar, true, 16>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 8:
				reduceRowSumKernel<T, Scalar, true, 8>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 4:
				reduceRowSumKernel<T, Scalar, true, 4>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 2:
				reduceRowSumKernel<T, Scalar, true, 2>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 1:
				reduceRowSumKernel<T, Scalar, true, 1>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			}
		}
		else {
			switch (threads) {
			case 1024:
				reduceRowSumKernel<T, Scalar, false, 1024>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 512:
				reduceRowSumKernel<T, Scalar, false, 512>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 256:
				reduceRowSumKernel<T, Scalar, false, 256>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 128:
				reduceRowSumKernel<T, Scalar, false, 128>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 64:
				reduceRowSumKernel<T, Scalar, false, 64>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 32:
				reduceRowSumKernel<T, Scalar, false, 32>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 16:
				reduceRowSumKernel<T, Scalar, false, 16>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 8:
				reduceRowSumKernel<T, Scalar, false, 8>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 4:
				reduceRowSumKernel<T, Scalar, false, 4>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 2:
				reduceRowSumKernel<T, Scalar, false, 2>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			case 1:
				reduceRowSumKernel<T, Scalar, false, 1>
					<< <gridSize, blockSize, sh_memSize, stream >> > (
						rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
				break;
			}
		}
	}

	/*
	 * 将 A 和 B 分块，每个 block 处理一部分的求和值，
	 * 即 A 被分为 rowElems 个行方向的 block，B 被分为 n(x_gridSize) 个列方向的
	 * block 每个行方向的 block大小为 x_blockSize，每个列方向的 block 大小为
	 * y_blockSize
	 * 每个行方向的 block，
	 * 都计算对应的 \sum func(ai, bj)(i:1->x_blockSize，j:1->y_blockSize)
	 * 最后得到 d_tRowSumMatrix(rowElems, n)
	 */
	template <typename T = Eigen::Vector3d, typename Scalar = double, bool useThrust = true>
	void execMyReduce(const cudaDeviceProp& prop, const cudaStream_t& stream,
		const int& rowElems, const uint& cols, const uint& paddingCols, const thrust::pair<T, uint32_t>* d_nodeVertexArray,
		const T* d_nodeWidthArray, const T* d_A, const Scalar* d_B, thrust::device_vector<Scalar>& d_value)
	{
		int x_blockSize = 0, y_blockSize = 64; // x操纵B，y操纵A
		int x_gridSize = 0, y_gridSize = (rowElems + y_blockSize - 1) / y_blockSize;

		// 分配时需要paddingCols
		getBlocksAndThreadsNum(prop, paddingCols, 65535, 1024 / y_blockSize, x_gridSize, x_blockSize);
		dim3 blockSize(x_blockSize, y_blockSize, 1);
		dim3 gridSize(x_gridSize, y_gridSize, 1);

		unsigned int x_paddingGridSize = PADDING_TO_WARP(x_gridSize);
		unsigned int t_rowSumMatrixSize = rowElems * x_paddingGridSize; // 分配时需要padding后的cols，单纯是为了用于后续重复计算row reduce sum

		thrust::device_vector<Scalar> d_tRowSumMatrix(t_rowSumMatrixSize, (Scalar).0);
		int sh_memSize = sizeof(Scalar) * y_blockSize * ((x_blockSize / 32) + 1); // +1 for avoiding bank conflicts
		bool flag = isPow2(cols);

		// d_tRowSumMatrix 为 row reduce sum 的结果，其实际不含 0 的数据维度为: elems * x_gridSize，
		// 而不是 elems * x_paddingGridSize
		switchKernel<T, Scalar>(flag, x_blockSize, gridSize, blockSize, sh_memSize, stream, rowElems,
			cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix.data().get());
		getLastCudaError("Kernel: 'reduceRowSumKernel' execution failed");
		//cudaDeviceSynchronize();

		int resCols = x_gridSize;
		if (resCols > 1) {
			thrust::device_vector<Scalar> rowSums(rowElems);
			if (useThrust)
				launch_ThrustRowSumReduce(rowElems, resCols, d_tRowSumMatrix, rowSums, stream);
			else
				launch_BLASRowSumReduce(rowElems, resCols, d_tRowSumMatrix.data().get(), rowSums.data().get(), stream);
			d_value = rowSums;
		}
		else {
			CUDA_CHECK(cudaMemcpyAsync(d_value.data().get(), d_tRowSumMatrix.data().get(),
				sizeof(Scalar) * rowElems, cudaMemcpyDeviceToDevice, stream));
		}
	}

	// argument is in host
	// single point
	void cpBSplineVal(const uint& numNodeVerts, const uint& numNodes,
		const V3d& pointData, const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>>& nodeVertexArray,
		const std::vector<V3d>& nodeWidthArray, const VXd& lambda, double& bSplinVal, const bool& useThrust)
	{
		// device
		cudaDeviceProp prop;
		int device = getMaxComputeDevice();
		CUDA_CHECK(cudaGetDevice(&device));
		CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

		thrust::pair<Eigen::Vector3d, uint32_t>* d_nodeVertexArray = nullptr;
		V3d* d_nodeWidthArray = nullptr;
		double* d_lambda = nullptr;
		V3d* d_pointData = nullptr;

		CUDA_CHECK(cudaMalloc((void**)&d_nodeVertexArray, sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts));
		CUDA_CHECK(cudaMalloc((void**)&d_nodeWidthArray, sizeof(V3d) * numNodes));
		CUDA_CHECK(cudaMalloc((void**)&d_lambda, sizeof(double) * lambda.rows()));
		CUDA_CHECK(cudaMalloc((void**)&d_pointData, sizeof(V3d)));

		CUDA_CHECK(cudaMemcpy(d_nodeVertexArray, nodeVertexArray.data(), sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_nodeWidthArray, nodeWidthArray.data(), sizeof(V3d) * numNodes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_lambda, lambda.data(), sizeof(double) * lambda.rows(), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_pointData, &pointData, sizeof(V3d), cudaMemcpyHostToDevice));

		thrust::device_vector<double> d_value(1);
		unsigned int paddingCols = PADDING_TO_WARP(numNodeVerts);
		if (useThrust) execMyReduce<V3d, double, true>(prop, 0, 1, numNodeVerts, paddingCols, d_nodeVertexArray, d_nodeWidthArray, d_pointData, d_lambda, d_value);
		else execMyReduce<V3d, double, false>(prop, 0, 1, numNodeVerts, paddingCols, d_nodeVertexArray, d_nodeWidthArray, d_pointData, d_lambda, d_value);
		
		CUDA_CHECK(cudaMemcpy(&bSplinVal, d_value.data().get(), sizeof(double), cudaMemcpyDeviceToHost));

		CUDA_CHECK(cudaFree(d_nodeVertexArray));
		CUDA_CHECK(cudaFree(d_nodeWidthArray));
		CUDA_CHECK(cudaFree(d_lambda));
		CUDA_CHECK(cudaFree(d_pointData));
		//free(h_bSplineVal);
	}

	void cpBSplineVal(const uint& numPoints, const uint& numNodeVerts, const uint& numNodes,
		const std::vector<V3d>& pointsData, const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>>& nodeVertexArray,
		const std::vector<V3d>& nodeWidthArray, const VXd& lambda, VXd& bSplinVal, const bool& useThrust)
	{
		// streams
		constexpr int MAX_NUM_STREAMS = 32;

		// device
		cudaDeviceProp prop;
		int device = getMaxComputeDevice();
		CUDA_CHECK(cudaGetDevice(&device));
		CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

		double* h_bSplineVal = nullptr;

		thrust::pair<Eigen::Vector3d, uint32_t>* d_nodeVertexArray = nullptr;
		V3d* d_nodeWidthArray = nullptr;
		double* d_lambda = nullptr;

		CUDA_CHECK(cudaHostAlloc((void**)&h_bSplineVal, sizeof(double) * numPoints, cudaHostAllocDefault));

		CUDA_CHECK(cudaMalloc((void**)&d_nodeVertexArray, sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts));
		CUDA_CHECK(cudaMalloc((void**)&d_nodeWidthArray, sizeof(V3d) * numNodes));
		unsigned int paddingCols = PADDING_TO_WARP(numNodeVerts);
		CUDA_CHECK(cudaMalloc((void**)&d_lambda, sizeof(double) * lambda.rows()));

		CUDA_CHECK(cudaMemcpy(d_nodeVertexArray, nodeVertexArray.data(), sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_nodeWidthArray, nodeWidthArray.data(), sizeof(V3d) * numNodes, cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(d_lambda, lambda.data(), sizeof(double) * lambda.rows(), cudaMemcpyHostToDevice));

		cudaStream_t streams[MAX_NUM_STREAMS];
		for (int i = 0; i < MAX_NUM_STREAMS; ++i) CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

		for (int i = 0; i < MAX_NUM_STREAMS; ++i) {
			int points_elems = (numPoints + MAX_NUM_STREAMS - 1) / MAX_NUM_STREAMS;
			int points_offset = i * points_elems;
			points_elems = points_offset + points_elems > numPoints ? numPoints - points_offset : points_elems;

			V3d* d_pointsData = nullptr;
			CUDA_CHECK(cudaMalloc((void**)&d_pointsData, sizeof(V3d) * points_elems));
			CUDA_CHECK(cudaMemcpyAsync(d_pointsData, pointsData.data() + points_offset, sizeof(V3d) * points_elems, cudaMemcpyHostToDevice, streams[i]));

			thrust::device_vector<double> d_value(points_elems);

			if (useThrust) execMyReduce<V3d, double, true>(prop, streams[i], points_elems, numNodeVerts, paddingCols, d_nodeVertexArray, d_nodeWidthArray, d_pointsData, d_lambda, d_value);
			else execMyReduce<V3d, double, false>(prop, streams[i], points_elems, numNodeVerts, paddingCols, d_nodeVertexArray, d_nodeWidthArray, d_pointsData, d_lambda, d_value);

			CUDA_CHECK(cudaMemcpyAsync(h_bSplineVal + points_offset, d_value.data().get(), sizeof(double) * points_elems, cudaMemcpyDeviceToHost, streams[i]));

			CUDA_CHECK(cudaFree(d_pointsData));
			cleanupThrust(d_value);
		}

		for (int i = 0; i < MAX_NUM_STREAMS; i++)
			cudaStreamSynchronize(streams[i]);
		for (int i = 0; i < MAX_NUM_STREAMS; ++i)
			CUDA_CHECK(cudaStreamDestroy(streams[i]));

		bSplinVal = Eigen::Map<Eigen::VectorXd>(h_bSplineVal, numPoints);

		CUDA_CHECK(cudaFree(d_nodeVertexArray));
		CUDA_CHECK(cudaFree(d_nodeWidthArray));
		CUDA_CHECK(cudaFree(d_lambda));
		//free(h_bSplineVal);
	}

	// argument is in device
	void cpBSplineVal(const cudaDeviceProp& prop, const uint& numPoints, const uint& numNodeVerts,
		const thrust::pair<Eigen::Vector3d, uint32_t>* d_nodeVertexArray, const V3d* d_nodeWidthArray,
		const double* d_lambda, const thrust::device_vector<V3d>& d_pointsData,
		thrust::device_vector<double>& d_bSplineVal, const cudaStream_t& stream, const bool& useThrust)
	{
		unsigned int paddingCols = PADDING_TO_WARP(numNodeVerts);

		if (useThrust) execMyReduce<V3d, double, true>(prop, stream, numPoints, numNodeVerts, paddingCols, d_nodeVertexArray, d_nodeWidthArray, d_pointsData.data().get(), d_lambda, d_bSplineVal);
		else execMyReduce<V3d, double, false>(prop, stream, numPoints, numNodeVerts, paddingCols, d_nodeVertexArray, d_nodeWidthArray, d_pointsData.data().get(), d_lambda, d_bSplineVal);
	}
}