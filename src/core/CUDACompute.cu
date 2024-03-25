#include "CUDACompute.hpp"
#include "MortonLUT.hpp"
#include "BSpline.hpp"
#include "utils/Common.hpp"
#include "detail/cuda/CUDAUtil.cuh"
#include "detail/cuda/DeviceQuery.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <thrust\reduce.h>
#include <thrust\generate.h>
#include <thrust\device_vector.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

//__global__ void intersectionKernel(const MatrixXd d_V,
//	const V2i* d_modelEdges,
//	const OctreeNode** d_leafNodes,
//	const int edgesNum,
//	const int leafNodesNum,
//	Vector3d* d_intersections)
//{
//	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
//	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
//
//	if (tx < edgesNum && ty < leafNodesNum)
//	{
//		Eigen::Vector2i e = d_modelEdges[tx];
//		Vector3d p1 = d_V.row(e.x());
//		Vector3d dir = d_V.row(e.y()) - p1;
//
//		auto node = d_leafNodes[ty];
//
//		Vector3d leftBottomBackCorner = node->boundary.first;
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

NAMESPACE_BEGIN(ITS)
    namespace cg = cooperative_groups;

    namespace cuAcc {

        // ���������η��ߺ����
        template<typename Real>
        __global__ void modelTriAttributeKernel(const size_t nTriangles,
                                                Triangle<Real> *d_modelTriangleArray) {
            const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

            if (tid < nTriangles) {
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
        void launch_modelTriAttributeKernel(const size_t &nTriangles,
                                            std::vector<Triangle<Eigen::Vector3d>> &modelTriangleArray) {
            thrust::device_vector<Triangle<Eigen::Vector3d>> d_modelTriangleArray = modelTriangleArray;
            int blockSize, gridSize, minGridSize;
            getOccupancyMaxPotentialBlockSize(nTriangles, minGridSize, blockSize, gridSize,
                                              modelTriAttributeKernel<Eigen::Vector3d>, 0, 0);
            modelTriAttributeKernel<Eigen::Vector3d> << <
            gridSize, blockSize >> > (nTriangles, d_modelTriangleArray.data().get());
            getLastCudaError("Kernel 'modelTriAttributeKernel' launch failed!\n");

            CUDA_CHECK(cudaMemcpy(modelTriangleArray.data(), d_modelTriangleArray.data().get(),
                                  sizeof(Triangle<Eigen::Vector3d>) * nTriangles, cudaMemcpyDeviceToHost));
        }

        //template <typename T>
        void launch_BLASRowSumReduce(const int &rows,
                                     const int &columns,
                                     double *d_matrix,
                                     double *d_row_sums,
                                     const cudaStream_t &stream) {
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

            std::vector<double> identityVector(columns, (double) 1.0); // ȫ1������
            double *d_vec;
            CUDA_CHECK(cudaMalloc((void **) &d_vec, sizeof(double) * columns));
            CUDA_CHECK(cudaMemcpyAsync(d_vec, identityVector.data(), sizeof(double) * columns, cudaMemcpyHostToDevice,
                                       stream));

            CUBLAS_CHECK(
                    cublasDgemm(cublasH, transa, transb, 1, rows, columns, &alpha, d_vec, lda, d_matrix, ldb, &beta,
                                d_row_sums, ldc));

            CUBLAS_CHECK(cublasDestroy(cublasH));
            CUDA_CHECK(cudaFree(d_vec));
        }

        // convert a linear index to a row index
        template<typename T>
        struct linear_index_to_row_index : public thrust::unary_function<T, T> {
            T C; // number of columns

            __host__ __device__
            linear_index_to_row_index(T C) : C(C) {}

            __host__ __device__
            T operator()(T i) {
                return i / C;
            }
        };

        // sum reduce at warp level
        template<typename T>
        __device__ __forceinline__ void warpReduceSum(unsigned int mask, T &sum) {
            for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(mask, sum, offset);
        }

        //template <typename T>
        void launch_ThrustRowSumReduce(const int &rows,
                                       const int &columns,
                                       const thrust::device_vector<double> &d_matrix,
                                       thrust::device_vector<double> &row_sums,
                                       const cudaStream_t &stream) // thrust::universal_vector
        {
            if (row_sums.size() != rows) {
                row_sums.clear();
                row_sums.resize(rows);
            }
//		assert(row_sums.size() == rows, "THE SIZE OF ROW_SUMS IS NOT EQUAL ROWS!\n");

            thrust::device_vector<int> row_indices(rows);

            if (stream)
                thrust::reduce_by_key(thrust::cuda::par.on(stream),
                                      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                                      linear_index_to_row_index<int>(columns)),
                                      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                                      linear_index_to_row_index<int>(rows)) +
                                      (rows * columns),
                                      d_matrix.begin(),
                                      row_indices.begin(),
                                      row_sums.begin(),
                                      thrust::equal_to<int>(),
                                      thrust::plus<double>());
            else
                thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                                      linear_index_to_row_index<int>(columns)),
                                      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                                      linear_index_to_row_index<int>(columns)) +
                                      (rows * columns),
                                      d_matrix.begin(),
                                      row_indices.begin(),
                                      row_sums.begin(),
                                      thrust::equal_to<int>(),
                                      thrust::plus<double>());
        }

        //__device__ void cpNumEdgeInterPoints(const V2i& e, const Vector3d& p1, const Vector3d& p2,
        //	const Vector3d& lbbCorner, const double& width, int& numInterPoints)
        //{
        //	Vector3d modelEdgeDir = p2 - p1;
        //
        //	// back plane
        //	double back_t = DINF;
        //	if (modelEdgeDir.x() != 0)
        //		back_t = (lbbCorner.x() - p1.x()) / modelEdgeDir.x();
        //	// left plane
        //	double left_t = DINF;
        //	if (modelEdgeDir.y() != 0)
        //		left_t = (lbbCorner.y() - p1.y()) / modelEdgeDir.y();
        //	// bottom plane
        //	double bottom_t = DINF;
        //	if (modelEdgeDir.z() != 0)
        //		bottom_t = (lbbCorner.z() - p1.z()) / modelEdgeDir.z();
        //
        //	if (isInRange(.0, 1.0, back_t) &&
        //		isInRange(lbbCorner.y(), lbbCorner.y() + width, (p1 + back_t * modelEdgeDir).y()) &&
        //		isInRange(lbbCorner.z(), lbbCorner.z() + width, (p1 + back_t * modelEdgeDir).z()))
        //	{
        //		++numInterPoints;
        //	}
        //	if (isInRange(.0, 1.0, left_t) &&
        //		isInRange(lbbCorner.x(), lbbCorner.x() + width, (p1 + left_t * modelEdgeDir).x()) &&
        //		isInRange(lbbCorner.z(), lbbCorner.z() + width, (p1 + left_t * modelEdgeDir).z()))
        //	{
        //		++numInterPoints;
        //	}
        //	if (isInRange(.0, 1.0, bottom_t) &&
        //		isInRange(lbbCorner.x(), lbbCorner.x() + width, (p1 + bottom_t * modelEdgeDir).x()) &&
        //		isInRange(lbbCorner.y(), lbbCorner.y() + width, (p1 + bottom_t * modelEdgeDir).y()))
        //	{
        //		++numInterPoints;
        //	}
        //}

        //template <typename T = Eigen::Vector3d, typename Scalar = double, bool nIsPow2, unsigned int colBlockSize>
        //__global__ void edgeIntersectionKernel(const uint nModelEdges,
        //	const V2i* d_modelEdgesArray, const Vector3d* d_modelVertsArray,
        //	const size_t numFineNodes, const Vector3d* d_nodeOriginArray, const double* d_nodeWidthArray,
        //	size_t* d_numEdgeInterPointsArray)
        //{
        //	int* shData = SharedMemory<int>();
        //	cg::thread_block ctb = cg::this_thread_block();
        //
        //	const unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
        //
        //	if (ty < nModelEdges)
        //	{
        //		unsigned int x_tid = threadIdx.x;
        //		unsigned int x_gridSize = colBlockSize * gridDim.x;
        //
        //		unsigned int maskLength = (colBlockSize & 31);
        //		maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
        //		const unsigned int mask = (0xffffffff) >> maskLength;
        //
        //		int numInterPoints = 0;
        //
        //		const V2i e = d_modelEdgesArray[ty];
        //		const Vector3d p1 = d_modelVertsArray[e.x()], p2 = d_modelVertsArray[e.y()];
        //
        //		if (nIsPow2)
        //		{
        //			unsigned int i = blockIdx.x * colBlockSize * 2 + threadIdx.x;
        //			x_gridSize <<= 1;
        //
        //			while (i < numFineNodes)
        //			{
        //				Vector3d lbbCorner = d_nodeOriginArray[i];
        //				double width = d_nodeWidthArray[i];
        //				cpNumEdgeInterPoints(e, p1, p2, lbbCorner, width, numInterPoints);
        //
        //				if (i + colBlockSize < numFineNodes)
        //				{
        //					lbbCorner = d_nodeOriginArray[i + colBlockSize];
        //					width = d_nodeWidthArray[i + colBlockSize];
        //					cpNumEdgeInterPoints(e, p1, p2, lbbCorner, width, numInterPoints);
        //
        //					i += x_gridSize;
        //				}
        //			}
        //		}
        //		else
        //		{
        //			unsigned int i = blockIdx.x * colBlockSize + threadIdx.x;
        //
        //			while (i < numFineNodes)
        //			{
        //				Vector3d lbbCorner = d_nodeOriginArray[i];
        //				double width = d_nodeWidthArray[i];
        //				cpNumEdgeInterPoints(e, p1, p2, lbbCorner, width, numInterPoints);
        //
        //				i += x_gridSize;
        //			}
        //		}
        //		// ��ÿ��warpִ�й�Լ��ͣ�Ȼ�󱣴浽shared memory��
        //		warpReduceSum<int>(mask, numInterPoints);
        //		const int sh_reduceNum = (colBlockSize / warpSize) > 0 ? colBlockSize / warpSize : 1;
        //		if (x_tid % warpSize == 0)
        //			shData[threadIdx.y * sh_reduceNum + x_tid / warpSize] = numInterPoints;
        //
        //		cg::sync(ctb);
        //
        //		const unsigned int newMask = __ballot_sync(mask, x_tid < sh_reduceNum);
        //		if (x_tid < sh_reduceNum) {
        //			numInterPoints = shData[threadIdx.y * sh_reduceNum + x_tid];
        //			warpReduceSum<int>(newMask, numInterPoints);
        //		}
        //
        //		if (x_tid == 0)
        //			d_numEdgeInterPointsArray[ty] = numInterPoints;
        //	}
        //}

        //__device__ void cpNumFaceInterPoints(const thrust::pair<Eigen::Vector3d, Eigen::Vector3d>& edge,
        //	const Vector3d& triEdge_1, const Vector3d& triEdge_2, const Vector3d& triEdge_3, const Vector3d& triNormal,
        //	const Vector3d& p1, const Vector3d& p2, const Vector3d& p3, const double& triDir, int& numInterPoints)
        //{
        //	Vector3d edgeDir = edge.second - edge.first;
        //
        //	if (fabsf(triNormal.dot(edgeDir)) < 1e-9) return;
        //
        //	double t = (-triDir - triNormal.dot(edge.first)) / (triNormal.dot(edgeDir));
        //	if (t < 0. || t > 1.) return;
        //	Vector3d interPoint = edge.first + edgeDir * t;
        //
        //	if (triEdge_1.cross(interPoint - p1).dot(triNormal) < 0) return;
        //	if (triEdge_2.cross(interPoint - p2).dot(triNormal) < 0) return;
        //	if (triEdge_3.cross(interPoint - p3).dot(triNormal) < 0) return;
        //
        //	++numInterPoints;
        //}

        //template <typename T = Eigen::Vector3d, typename Scalar = double, bool nIsPow2, unsigned int colBlockSize>
        //__global__ void faceIntersectionKernel(const uint nModelTris, const Triangle<Vector3d>* d_modelTrisArray,
        //	const uint numFineNodeEdges, const thrust::pair<thrust::pair<Vector3d, Vector3d>, uint32_t>* d_fineNodeEdgesArray,
        //	size_t* d_numFaceInterPointsArray)
        //{
        //	int* shData = SharedMemory<int>();
        //	cg::thread_block ctb = cg::this_thread_block();
        //
        //	const unsigned int ty = threadIdx.y + blockIdx.y * blockDim.y;
        //
        //	if (ty < nModelTris)
        //	{
        //		unsigned int x_tid = threadIdx.x;
        //		unsigned int x_gridSize = colBlockSize * gridDim.x;
        //
        //		unsigned int maskLength = (colBlockSize & 31);
        //		maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
        //		const unsigned int mask = (0xffffffff) >> maskLength;
        //
        //		int numInterPoints = 0;
        //
        //		Triangle<Vector3d> tri = d_modelTrisArray[ty];
        //		Vector3d triEdge_1 = tri.p2 - tri.p1; Vector3d triEdge_2 = tri.p3 - tri.p2; Vector3d triEdge_3 = tri.p1 - tri.p3;
        //		Vector3d triNormal = tri.normal; double triDir = tri.dir;
        //
        //		if (nIsPow2)
        //		{
        //			unsigned int i = blockIdx.x * colBlockSize * 2 + threadIdx.x;
        //			x_gridSize <<= 1;
        //
        //			while (i < numFineNodeEdges)
        //			{
        //				thrust::pair<thrust::pair<Vector3d, Vector3d>, uint32_t> nodeEdge = d_fineNodeEdgesArray[i];
        //				cpNumFaceInterPoints(nodeEdge.first, triEdge_1, triEdge_2, triEdge_3, triNormal, tri.p1, tri.p2, tri.p3, triDir, numInterPoints);
        //
        //				if (i + colBlockSize < numFineNodes)
        //				{
        //					thrust::pair<thrust::pair<Vector3d, Vector3d>, uint32_t> nodeEdge = d_fineNodeEdgesArray[i + colBlockSize];
        //					cpNumFaceInterPoints(nodeEdge.first, triEdge_1, triEdge_2, triEdge_3, triNormal, tri.p1, tri.p2, tri.p3, triDir, numInterPoints);
        //
        //					i += x_gridSize;
        //				}
        //			}
        //		}
        //		else
        //		{
        //			unsigned int i = blockIdx.x * colBlockSize + threadIdx.x;
        //
        //			while (i < numFineNodeEdges)
        //			{
        //				thrust::pair<thrust::pair<Vector3d, Vector3d>, uint32_t> nodeEdge = d_fineNodeEdgesArray[i];
        //				cpNumFaceInterPoints(nodeEdge.first, triEdge_1, triEdge_2, triEdge_3, triNormal, tri.p1, tri.p2, tri.p3, triDir, numInterPoints);
        //
        //				i += x_gridSize;
        //			}
        //		}
        //		// ��ÿ��warpִ�й�Լ��ͣ�Ȼ�󱣴浽shared memory��
        //		warpReduceSum<int>(mask, numInterPoints);
        //		const int sh_reduceNum = (colBlockSize / warpSize) > 0 ? colBlockSize / warpSize : 1;
        //		if (x_tid % warpSize == 0)
        //			shData[threadIdx.y * sh_reduceNum + x_tid / warpSize] = numInterPoints;
        //
        //		cg::sync(ctb);
        //
        //		const unsigned int newMask = __ballot_sync(mask, x_tid < sh_reduceNum);
        //		if (x_tid < sh_reduceNum) {
        //			numInterPoints = shData[threadIdx.y * sh_reduceNum + x_tid];
        //			warpReduceSum<int>(newMask, numInterPoints);
        //		}
        //
        //		if (x_tid == 0)
        //			d_numFaceInterPointsArray[ty] = numInterPoints;
        //	}
        //
        //}

        //void cpIntersection(const uint& nModelEdges, const vector<V2i>& modelEdgesArray, const vector<Vector3d> modelVertsArray,
        //	const size_t& numFineNodes, const vector<Vector3d>& nodeOriginArray, const vector<double>& nodeWidthArray,
        //	const uint& nModelTris, const vector<Triangle<Vector3d>>& modelTrisArray,
        //	const uint& numFineNodeEdges, const vector<thrust::pair<thrust::pair<Vector3d, Vector3d>, uint32_t>>& fineNodeEdgesArray)
        //{
        //	constexpr int MAX_STREAM = 2;
        //	cudaStream_t streams[MAX_STREAM];
        //	for (int i = 0; i < MAX_STREAM; ++i) CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
        //
        //	dim3 gridSize, blockSize;
        //
        //	thrust::device_vector<size_t> d_numEdgeInterPointsArray;
        //
        //
        //
        //	for (int i = 0; i < MAX_STREAM; ++i) CUDA_CHECK(cudaStreamDestroy(streams[i]));
        //}

        /*
         * matrix reduce for sum of row
         * @param m: rows
         * @param n: columns, and n % warpSize = 0
         * @param g_idata: matrix(m * n)
         */
        template<typename T = Eigen::Vector3d, typename Scalar = double, bool nIsPow2, unsigned int colBlockSize>
        __global__ void reduceRowSumKernel(const unsigned int m, const unsigned int n,
                                           const thrust::pair<T, uint32_t> *__restrict__ d_nodeVertexArray,
                                           const T *__restrict__ d_nodeWidthArray,
                                           const T *__restrict__ g_iA,
                                           const Scalar *__restrict__ g_iB, // lambda
                                           Scalar *__restrict__ g_odata) {
            Scalar *shData = SharedMemory<Scalar>();
            cg::thread_block ctb = cg::this_thread_block();

            unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;

            if (ty < m) {
                unsigned int x_tid = threadIdx.x;
                unsigned int x_gridSize = colBlockSize * gridDim.x;

                unsigned int maskLength = (colBlockSize & 31);
                maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
                const unsigned int mask = (0xffffffff) >> maskLength;

                Scalar sum = (Scalar) .0;

                //printf("#1 sum = %lf\n", sum);
                // reduce multiple elements per thread
                if (nIsPow2) {
                    unsigned int i = blockIdx.x * colBlockSize * 2 + threadIdx.x;
                    x_gridSize <<= 1;

                    while (i < n) {
                        //printf("#2 sum = %lf\n", sum);
                        sum += g_iB[i] * core::BaseFunction4Point(d_nodeVertexArray[i].first,
                                                            d_nodeWidthArray[d_nodeVertexArray[i].second], g_iA[ty]);
                        if (i + colBlockSize < n) {
                            sum += g_iB[i + colBlockSize] *
                                   core::BaseFunction4Point(d_nodeVertexArray[i + colBlockSize].first,
                                                      d_nodeWidthArray[d_nodeVertexArray[i + colBlockSize].second],
                                                      g_iA[ty]); // (һ��)�߳̿鼶��Ŀ��
                            i += x_gridSize; // ���񼶱�Ŀ�ȣ�Ĭ�������С(block������)Ϊԭ������(xά�ȼ�����)��һ��(���nIsPow2��������x_gridSize����һ��)
                        }
                    }
                } else {
                    unsigned int i = blockIdx.x * colBlockSize + threadIdx.x;
                    while (i < n) {
                        //printf("#2 sum = %lf\n", sum);
                        sum += g_iB[i] * core::BaseFunction4Point(d_nodeVertexArray[i].first,
                                                            d_nodeWidthArray[d_nodeVertexArray[i].second], g_iA[ty]);
                        i += x_gridSize;
                    }
                }

                // ��ÿ��warpִ�й�Լ��ͣ�Ȼ�󱣴浽shared memory��
                warpReduceSum<Scalar>(mask, sum);
                const int sh_reduceNum = (colBlockSize / warpSize) > 0 ? colBlockSize / warpSize : 1;
                if (x_tid % warpSize == 0)
                    shData[threadIdx.y * sh_reduceNum + x_tid / warpSize] = sum;

                cg::sync(ctb);

                //printf("#3 sum = %lf\n", sum);

                // ͬһ��block������warp���(ֻҪ��ÿ��warp�ĵ�һ��thread�����sum���������ɣ�
                // ��Ϊÿ��warp�ĵ�һ��thread�����sum����������warp�������̵߳����ݺ�)
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

        template<typename T = Eigen::Vector3d, typename Scalar = double, bool nIsPow2, unsigned int colBlockSize>
        __global__ void mq_reduceRowSumKernel(const unsigned int m, const unsigned int n,
                                              const thrust::pair<T, uint32_t> *__restrict__ d_nodeVertexArray,
                                              const T *__restrict__ d_nodeWidthArray,
                                              const T *__restrict__ g_iA,
                                              const Scalar *__restrict__ g_iB, // lambda
                                              const Scalar *__restrict__ d_outerVal,
                                              const Eigen::Array3d *__restrict__ d_minRange,
                                              const Eigen::Array3d *__restrict__ d_maxRange,
                                              Scalar *__restrict__ g_odata) {
            Scalar *shData = SharedMemory<Scalar>();
            cg::thread_block ctb = cg::this_thread_block();

            unsigned int ty = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int tx = blockIdx.x * blockDim.x + threadIdx.x;

            if (ty < m) {
                if ((g_iA[ty].array() < *d_minRange).any() || (g_iA[ty].array() > *d_maxRange).any()) {
                    if (tx == 0) g_odata[ty * gridDim.x] = *d_outerVal;
                    return;
                }

                unsigned int x_tid = threadIdx.x;
                unsigned int x_gridSize = colBlockSize * gridDim.x;

                unsigned int maskLength = (colBlockSize & 31);
                maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
                const unsigned int mask = (0xffffffff) >> maskLength;

                Scalar sum = (Scalar) .0;

                //printf("#1 sum = %lf\n", sum);
                // reduce multiple elements per thread
                if (nIsPow2) {
                    unsigned int i = blockIdx.x * colBlockSize * 2 + threadIdx.x;
                    x_gridSize <<= 1;

                    while (i < n) {
                        sum += g_iB[i] * core::BaseFunction4Point(d_nodeVertexArray[i].first,
                                                            d_nodeWidthArray[d_nodeVertexArray[i].second], g_iA[ty]);
                        if (i + colBlockSize < n) {
                            sum += g_iB[i + colBlockSize] *
                                   core::BaseFunction4Point(d_nodeVertexArray[i + colBlockSize].first,
                                                      d_nodeWidthArray[d_nodeVertexArray[i + colBlockSize].second],
                                                      g_iA[ty]); // (һ��)�߳̿鼶��Ŀ��
                            i += x_gridSize; // ���񼶱�Ŀ�ȣ�Ĭ�������С(block������)Ϊԭ������(xά�ȼ�����)��һ��(���nIsPow2��������x_gridSize����һ��)
                        }
                    }
                } else {
                    unsigned int i = blockIdx.x * colBlockSize + threadIdx.x;
                    while (i < n) {
                        sum += g_iB[i] * core::BaseFunction4Point(d_nodeVertexArray[i].first,
                                                            d_nodeWidthArray[d_nodeVertexArray[i].second], g_iA[ty]);
                        i += x_gridSize;
                    }
                }

                // ��ÿ��warpִ�й�Լ��ͣ�Ȼ�󱣴浽shared memory��
                warpReduceSum<Scalar>(mask, sum);
                const int sh_reduceNum = (colBlockSize / warpSize) > 0 ? colBlockSize / warpSize : 1;
                if (x_tid % warpSize == 0)
                    shData[threadIdx.y * sh_reduceNum + x_tid / warpSize] = sum;

                cg::sync(ctb);

                // ͬһ��block������warp���(ֻҪ��ÿ��warp�ĵ�һ��thread�����sum���������ɣ�
                // ��Ϊÿ��warp�ĵ�һ��thread�����sum����������warp�������̵߳����ݺ�)
                const unsigned int newMask = __ballot_sync(mask, x_tid < sh_reduceNum);
                if (x_tid < sh_reduceNum) {
                    sum = shData[threadIdx.y * sh_reduceNum + x_tid];
                    warpReduceSum<Scalar>(newMask, sum);
                }

                if (x_tid == 0) {
                    g_odata[ty * gridDim.x + blockIdx.x] = sum;
                }
            }
        }

        template<typename T = Eigen::Vector3d, typename Scalar = double>
        void switchKernel(const bool &isPow2, const int &threads, const dim3 &gridSize,
                          const dim3 &blockSize, const int &sh_memSize, const cudaStream_t &stream,
                          const uint &rowElems,
                          const uint &cols, const thrust::pair<T, uint32_t> *d_nodeVertexArray,
                          const T *d_nodeWidthArray, const T *d_A, const Scalar *d_B, Scalar *d_tRowSumMatrix) {
            if (isPow2) {
                switch (threads) {
                    case 1024:
                        reduceRowSumKernel<T, Scalar, true, 1024>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 512:
                        reduceRowSumKernel<T, Scalar, true, 512>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 256:
                        reduceRowSumKernel<T, Scalar, true, 256>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 128:
                        reduceRowSumKernel<T, Scalar, true, 128>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 64:
                        reduceRowSumKernel<T, Scalar, true, 64>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 32:
                        reduceRowSumKernel<T, Scalar, true, 32>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 16:
                        reduceRowSumKernel<T, Scalar, true, 16>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 8:
                        reduceRowSumKernel<T, Scalar, true, 8>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 4:
                        reduceRowSumKernel<T, Scalar, true, 4>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 2:
                        reduceRowSumKernel<T, Scalar, true, 2>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 1:
                        reduceRowSumKernel<T, Scalar, true, 1>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                }
            } else {
                switch (threads) {
                    case 1024:
                        reduceRowSumKernel<T, Scalar, false, 1024>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 512:
                        reduceRowSumKernel<T, Scalar, false, 512>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 256:
                        reduceRowSumKernel<T, Scalar, false, 256>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 128:
                        reduceRowSumKernel<T, Scalar, false, 128>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 64:
                        reduceRowSumKernel<T, Scalar, false, 64>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 32:
                        reduceRowSumKernel<T, Scalar, false, 32>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 16:
                        reduceRowSumKernel<T, Scalar, false, 16>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 8:
                        reduceRowSumKernel<T, Scalar, false, 8>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 4:
                        reduceRowSumKernel<T, Scalar, false, 4>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 2:
                        reduceRowSumKernel<T, Scalar, false, 2>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                    case 1:
                        reduceRowSumKernel<T, Scalar, false, 1>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_tRowSumMatrix);
                        break;
                }
            }
        }

        // ���ڶ���ѯ
        template<typename T = Eigen::Vector3d, typename Scalar = double>
        void mq_switchKernel(const bool &isPow2, const int &threads, const dim3 &gridSize,
                             const dim3 &blockSize, const int &sh_memSize, const cudaStream_t &stream,
                             const uint &rowElems,
                             const uint &cols, const thrust::pair<T, uint32_t> *d_nodeVertexArray,
                             const T *d_nodeWidthArray, const T *d_A, const Scalar *d_B, const Scalar *d_outerVal,
                             const Eigen::Array3d *d_minRange,
                             const Eigen::Array3d *d_maxRange, Scalar *d_tRowSumMatrix) {
            if (isPow2) {
                switch (threads) {
                    case 1024:
                        mq_reduceRowSumKernel<T, Scalar, true, 1024>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 512:
                        mq_reduceRowSumKernel<T, Scalar, true, 512>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 256:
                        mq_reduceRowSumKernel<T, Scalar, true, 256>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 128:
                        mq_reduceRowSumKernel<T, Scalar, true, 128>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 64:
                        mq_reduceRowSumKernel<T, Scalar, true, 64>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 32:
                        mq_reduceRowSumKernel<T, Scalar, true, 32>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 16:
                        mq_reduceRowSumKernel<T, Scalar, true, 16>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 8:
                        mq_reduceRowSumKernel<T, Scalar, true, 8>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 4:
                        mq_reduceRowSumKernel<T, Scalar, true, 4>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 2:
                        mq_reduceRowSumKernel<T, Scalar, true, 2>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 1:
                        mq_reduceRowSumKernel<T, Scalar, true, 1>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                }
            } else {
                switch (threads) {
                    case 1024:
                        mq_reduceRowSumKernel<T, Scalar, false, 1024>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 512:
                        mq_reduceRowSumKernel<T, Scalar, false, 512>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 256:
                        mq_reduceRowSumKernel<T, Scalar, false, 256>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 128:
                        mq_reduceRowSumKernel<T, Scalar, false, 128>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 64:
                        mq_reduceRowSumKernel<T, Scalar, false, 64>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 32:
                        mq_reduceRowSumKernel<T, Scalar, false, 32>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 16:
                        mq_reduceRowSumKernel<T, Scalar, false, 16>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 8:
                        mq_reduceRowSumKernel<T, Scalar, false, 8>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 4:
                        mq_reduceRowSumKernel<T, Scalar, false, 4>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 2:
                        mq_reduceRowSumKernel<T, Scalar, false, 2>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                    case 1:
                        mq_reduceRowSumKernel<T, Scalar, false, 1>
                                << < gridSize, blockSize, sh_memSize, stream >> > (
                        rowElems, cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange, d_maxRange, d_tRowSumMatrix);
                        break;
                }
            }
        }

        /*
         * �� A �� B �ֿ飬ÿ�� block ����һ���ֵ����ֵ��
         * �� A ����Ϊ rowElems ���з���� block��B ����Ϊ n(x_gridSize) ���з����
         * block ÿ���з���� block��СΪ x_blockSize��ÿ���з���� block ��СΪ
         * y_blockSize
         * ÿ���з���� block��
         * �������Ӧ�� \sum func(ai, bj)(i:1->x_blockSize��j:1->y_blockSize)
         * ���õ� d_tRowSumMatrix(rowElems, n)
         */
        template<typename T = Vector3d, typename Scalar = double, bool useThrust = true>
        void execMyReduce(const cudaDeviceProp &prop, const cudaStream_t &stream,
                          const uint &rowElems, const uint &cols, const uint &paddingCols,
                          const thrust::pair<T, uint32_t> *d_nodeVertexArray,
                          const T *d_nodeWidthArray, const T *d_A, const Scalar *d_B,
                          thrust::device_vector<Scalar> &d_value) {
            //std::cout << "rowElems = " << rowElems << std::endl;
            int x_blockSize = 0, y_blockSize = 16; // x����B��y����A (ע�⣺x_blockSize �����>=32)
            int x_gridSize = 0, y_gridSize = (rowElems + y_blockSize - 1) / y_blockSize;

            // ����ʱ��ҪpaddingCols
            getBlocksAndThreadsNum(prop, paddingCols, 65535, 1024 / y_blockSize, x_gridSize, x_blockSize);
            dim3 blockSize(x_blockSize, y_blockSize, 1);
            dim3 gridSize(x_gridSize, y_gridSize, 1);

            unsigned int x_paddingGridSize = PADDING_TO_WARP(x_gridSize);
            unsigned int t_rowSumMatrixSize =
                    rowElems * x_paddingGridSize; // ����ʱ��Ҫpadding���cols��������Ϊ�����ں����ظ�����row reduce sum

            thrust::device_vector<Scalar> d_tRowSumMatrix(t_rowSumMatrixSize, (Scalar) .0);
            int sh_memSize = sizeof(Scalar) * y_blockSize * ((x_blockSize / 32) + 1); // +1 for avoiding bank conflicts
            bool flag = isPow2(cols);

            // d_tRowSumMatrix Ϊ row reduce sum �Ľ������ʵ�ʲ��� 0 ������ά��Ϊ: elems * x_gridSize��
            // ������ elems * x_paddingGridSize
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
                    launch_BLASRowSumReduce(rowElems, resCols, d_tRowSumMatrix.data().get(), rowSums.data().get(),
                                            stream);
                d_value = rowSums;
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_value.data().get(), d_tRowSumMatrix.data().get(),
                                           sizeof(Scalar) * rowElems, cudaMemcpyDeviceToDevice, stream));
            }
        }

        // ���ڶ���ѯ
        template<typename T = Eigen::Vector3d, typename Scalar = double, bool useThrust = true>
        void mq_execMyReduce(const cudaDeviceProp &prop, const cudaStream_t &stream,
                             const uint &rowElems, const uint &cols, const uint &paddingCols,
                             const thrust::pair<T, uint32_t> *d_nodeVertexArray,
                             const T *d_nodeWidthArray, const T *d_A, const Scalar *d_B, const Scalar *d_outerVal,
                             const Eigen::Array3d *d_minRange, const Eigen::Array3d *d_maxRange,
                             thrust::device_vector<Scalar> &d_value) {
            int x_blockSize = 0, y_blockSize = 32; // x����B��y����A (ע�⣺x_blockSize �����>=32)
            int x_gridSize = 0, y_gridSize = (rowElems + y_blockSize - 1) / y_blockSize;

            // ����ʱ��ҪpaddingCols
            getBlocksAndThreadsNum(prop, paddingCols, 65535, 1024 / y_blockSize, x_gridSize, x_blockSize);
            dim3 blockSize(x_blockSize, y_blockSize, 1);
            dim3 gridSize(x_gridSize, y_gridSize, 1);

            unsigned int x_paddingGridSize = PADDING_TO_WARP(x_gridSize);
            unsigned int t_rowSumMatrixSize =
                    rowElems * x_paddingGridSize; // ����ʱ��Ҫpadding���cols��������Ϊ�����ں����ظ�����row reduce sum

            thrust::device_vector<Scalar> d_tRowSumMatrix(t_rowSumMatrixSize, (Scalar) .0);
            int sh_memSize = sizeof(Scalar) * y_blockSize * ((x_blockSize / 32) + 1); // +1 for avoiding bank conflicts
            bool flag = isPow2(cols);

            // d_tRowSumMatrix Ϊ row reduce sum �Ľ������ʵ�ʲ��� 0 ������ά��Ϊ: elems * x_gridSize��
            // ������ elems * x_paddingGridSize
            mq_switchKernel<T, Scalar>(flag, x_blockSize, gridSize, blockSize, sh_memSize, stream, rowElems,
                                       cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B, d_outerVal, d_minRange,
                                       d_maxRange,
                                       d_tRowSumMatrix.data().get());
            getLastCudaError("Kernel: 'reduceRowSumKernel' execution failed");
            //cudaDeviceSynchronize();

            int resCols = x_gridSize;
            if (resCols > 1) {
                thrust::device_vector<Scalar> rowSums(rowElems);
                if (useThrust)
                    launch_ThrustRowSumReduce(rowElems, resCols, d_tRowSumMatrix, rowSums, stream);
                else
                    launch_BLASRowSumReduce(rowElems, resCols, d_tRowSumMatrix.data().get(), rowSums.data().get(),
                                            stream);
                d_value = rowSums;
            } else {
                CUDA_CHECK(cudaMemcpyAsync(d_value.data().get(), d_tRowSumMatrix.data().get(),
                                           sizeof(Scalar) * rowElems, cudaMemcpyDeviceToDevice, stream));
            }
        }

        // argument is in host
        // single point
        void cpBSplineVal(const uint &numNodeVerts, const uint &numNodes,
                          const Vector3d &pointData,
                          const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>> &nodeVertexArray,
                          const std::vector<Vector3d> &nodeWidthArray, const VectorXd &lambda, double &bSplinVal,
                          bool useThrust) {
            // device
            cudaDeviceProp prop;
            int device = getMaxComputeDevice();
            CUDA_CHECK(cudaGetDevice(&device));
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

            thrust::pair<Eigen::Vector3d, uint32_t> *d_nodeVertexArray = nullptr;
            Vector3d *d_nodeWidthArray = nullptr;
            double *d_lambda = nullptr;
            Vector3d *d_pointData = nullptr;

            CUDA_CHECK(cudaMalloc((void **) &d_nodeVertexArray,
                                  sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts));
            CUDA_CHECK(cudaMalloc((void **) &d_nodeWidthArray, sizeof(Vector3d) * numNodes));
            CUDA_CHECK(cudaMalloc((void **) &d_lambda, sizeof(double) * lambda.rows()));
            CUDA_CHECK(cudaMalloc((void **) &d_pointData, sizeof(Vector3d)));

            CUDA_CHECK(cudaMemcpy(d_nodeVertexArray, nodeVertexArray.data(),
                                  sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_nodeWidthArray, nodeWidthArray.data(), sizeof(Vector3d) * numNodes,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_lambda, lambda.data(), sizeof(double) * lambda.rows(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pointData, &pointData, sizeof(Vector3d), cudaMemcpyHostToDevice));

            thrust::device_vector<double> d_value(1);
            unsigned int paddingCols = PADDING_TO_WARP(numNodeVerts);
            if (useThrust)
                execMyReduce<Vector3d, double, true>(prop, 0, 1, numNodeVerts, paddingCols, d_nodeVertexArray,
                                                d_nodeWidthArray, d_pointData, d_lambda, d_value);
            else
                execMyReduce<Vector3d, double, false>(prop, 0, 1, numNodeVerts, paddingCols, d_nodeVertexArray,
                                                 d_nodeWidthArray, d_pointData, d_lambda, d_value);

            CUDA_CHECK(cudaMemcpy(&bSplinVal, d_value.data().get(), sizeof(double), cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaFree(d_nodeVertexArray));
            CUDA_CHECK(cudaFree(d_nodeWidthArray));
            CUDA_CHECK(cudaFree(d_lambda));
            CUDA_CHECK(cudaFree(d_pointData));
            //free(h_bSplineVal);
        }

        void cpBSplineVal(const uint &numPoints, const uint &numNodeVerts, const uint &numNodes,
                          const std::vector<Vector3d> &pointsData,
                          const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>> &nodeVertexArray,
                          const std::vector<Vector3d> &nodeWidthArray, const VectorXd &lambda, VectorXd &bSplinVal,
                          const bool &useThrust) {
            // streams
            constexpr int MAX_NUM_STREAMS = 32;
            static_assert(MAX_NUM_STREAMS >= 1, "THE NUMBER OF STREAMS IS INVALID");

            // device
            cudaDeviceProp prop;
            int device = getMaxComputeDevice();
            CUDA_CHECK(cudaGetDevice(&device));
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

            double *h_bSplineVal = nullptr;

            thrust::pair<Eigen::Vector3d, uint32_t> *d_nodeVertexArray = nullptr;
            Vector3d *d_nodeWidthArray = nullptr;
            double *d_lambda = nullptr;

            CUDA_CHECK(cudaHostAlloc((void **) &h_bSplineVal, sizeof(double) * numPoints, cudaHostAllocDefault));

            CUDA_CHECK(cudaMalloc((void **) &d_nodeVertexArray,
                                  sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts));
            CUDA_CHECK(cudaMalloc((void **) &d_nodeWidthArray, sizeof(Vector3d) * numNodes));
            unsigned int paddingCols = PADDING_TO_WARP(numNodeVerts);
            CUDA_CHECK(cudaMalloc((void **) &d_lambda, sizeof(double) * lambda.rows()));

            CUDA_CHECK(cudaMemcpy(d_nodeVertexArray, nodeVertexArray.data(),
                                  sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_nodeWidthArray, nodeWidthArray.data(), sizeof(Vector3d) * numNodes,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_lambda, lambda.data(), sizeof(double) * lambda.rows(), cudaMemcpyHostToDevice));

            cudaStream_t streams[MAX_NUM_STREAMS];
            for (int i = 0; i < MAX_NUM_STREAMS; ++i)
                CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

            for (int i = 0; i < MAX_NUM_STREAMS; ++i) {

                uint points_elems = (numPoints + MAX_NUM_STREAMS - 1) / MAX_NUM_STREAMS;
                uint points_offset = i * points_elems;
                bool lastBatch = false;
                if (points_offset + points_elems > numPoints) {
                    lastBatch = true;
                    points_elems = numPoints - points_offset;
                }

                Vector3d *d_pointsData = nullptr;
                CUDA_CHECK(cudaMalloc((void **) &d_pointsData, sizeof(Vector3d) * points_elems));
                CUDA_CHECK(cudaMemcpyAsync(d_pointsData, pointsData.data() + points_offset, sizeof(Vector3d) * points_elems,
                                           cudaMemcpyHostToDevice, streams[i]));

                thrust::device_vector<double> d_value(points_elems);

                if (useThrust)
                    execMyReduce<Vector3d, double, true>(prop, streams[i], points_elems, numNodeVerts, paddingCols,
                                                    d_nodeVertexArray, d_nodeWidthArray, d_pointsData, d_lambda,
                                                    d_value);
                else
                    execMyReduce<Vector3d, double, false>(prop, streams[i], points_elems, numNodeVerts, paddingCols,
                                                     d_nodeVertexArray, d_nodeWidthArray, d_pointsData, d_lambda,
                                                     d_value);

                CUDA_CHECK(cudaMemcpyAsync(h_bSplineVal + points_offset, d_value.data().get(),
                                           sizeof(double) * points_elems, cudaMemcpyDeviceToHost, streams[i]));

                CUDA_CHECK(cudaFree(d_pointsData));
                cleanupThrust(d_value);

                printf("[%d/%d] batch_size = %u", i + 1, MAX_NUM_STREAMS, points_elems);
                if (i != MAX_NUM_STREAMS - 1 && !lastBatch) printf("\r");
                else {
                    printf("\n");
                    break;
                }
            }

            for (int i = 0; i < MAX_NUM_STREAMS; i++)
                cudaStreamSynchronize(streams[i]);
            for (int i = 0; i < MAX_NUM_STREAMS; ++i)
                CUDA_CHECK(cudaStreamDestroy(streams[i]));

            bSplinVal = Eigen::Map<VectorXd>(h_bSplineVal, numPoints);

            CUDA_CHECK(cudaFree(d_nodeVertexArray));
            CUDA_CHECK(cudaFree(d_nodeWidthArray));
            CUDA_CHECK(cudaFree(d_lambda));
            //free(h_bSplineVal);
        }

        // argument is in device
        void cpBSplineVal(const cudaDeviceProp &prop, const uint &numPoints, const uint &numNodeVerts,
                          const thrust::pair<Eigen::Vector3d, uint32_t> *d_nodeVertexArray, const Vector3d *d_nodeWidthArray,
                          const double *d_lambda, const thrust::device_vector<Vector3d> &d_pointsData,
                          thrust::device_vector<double> &d_bSplineVal, const cudaStream_t &stream,
                          bool useThrust) {
            unsigned int paddingCols = PADDING_TO_WARP(numNodeVerts);

            if (useThrust)
                execMyReduce<Vector3d, double, true>(prop, stream, numPoints, numNodeVerts, paddingCols, d_nodeVertexArray,
                                                d_nodeWidthArray, d_pointsData.data().get(), d_lambda, d_bSplineVal);
            else
                execMyReduce<Vector3d, double, false>(prop, stream, numPoints, numNodeVerts, paddingCols, d_nodeVertexArray,
                                                 d_nodeWidthArray, d_pointsData.data().get(), d_lambda, d_bSplineVal);
        }

        void cpPointQuery(const uint &numPoints, const uint &numNodeVerts,
                          const uint &numNodes, const Array3d &minRange, const Array3d &maxRange,
                          const std::vector<Vector3d> &pointsData,
                          const std::vector<thrust::pair<Vector3d, uint32_t>> &nodeVertexArray,
                          const std::vector<Vector3d> &nodeWidthArray, const VectorXd &lambda, const double &outerVal,
                          VectorXd &bSplinVal, bool useThrust) {
            // streams
            constexpr int MAX_NUM_STREAMS = 4;
            static_assert(MAX_NUM_STREAMS >= 1, "THE NUMBER OF STREAMS IS INVALID");

            // device
            cudaDeviceProp prop;
            int device = getMaxComputeDevice();
            CUDA_CHECK(cudaGetDevice(&device));
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

            double *h_bSplineVal = nullptr;

            thrust::pair<Vector3d, uint32_t> *d_nodeVertexArray = nullptr;
            Vector3d *d_nodeWidthArray = nullptr;
            double *d_lambda = nullptr;
            double *d_outerVal = nullptr;
            // ���ڼ��������boundingbox��ԭ���b����ֵ����һ�������жϵ���ģ�͵����⣩
            Eigen::Array3d *d_minRange = nullptr;
            Eigen::Array3d *d_maxRange = nullptr;

            CUDA_CHECK(cudaHostAlloc((void **) &h_bSplineVal, sizeof(double) * numPoints, cudaHostAllocDefault));

            CUDA_CHECK(cudaMalloc((void **) &d_nodeVertexArray,
                                  sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts));
            CUDA_CHECK(cudaMalloc((void **) &d_nodeWidthArray, sizeof(Vector3d) * numNodes));
            unsigned int paddingCols = PADDING_TO_WARP(numNodeVerts);
            CUDA_CHECK(cudaMalloc((void **) &d_lambda, sizeof(double) * lambda.rows()));
            CUDA_CHECK(cudaMalloc((void **) &d_outerVal, sizeof(double)));
            CUDA_CHECK(cudaMalloc((void **) &d_minRange, sizeof(Vector3d)));
            CUDA_CHECK(cudaMalloc((void **) &d_maxRange, sizeof(Vector3d)));

            CUDA_CHECK(cudaMemcpy(d_nodeVertexArray, nodeVertexArray.data(),
                                  sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_nodeWidthArray, nodeWidthArray.data(), sizeof(Vector3d) * numNodes,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_lambda, lambda.data(), sizeof(double) * lambda.rows(), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_outerVal, &outerVal, sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_minRange, &minRange, sizeof(Eigen::Array3d), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_maxRange, &maxRange, sizeof(Eigen::Array3d), cudaMemcpyHostToDevice));

            cudaStream_t streams[MAX_NUM_STREAMS];
            for (int i = 0; i < MAX_NUM_STREAMS; ++i)
                CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

            for (int i = 0; i < MAX_NUM_STREAMS; ++i) {
                uint points_elems = (numPoints + MAX_NUM_STREAMS - 1) / MAX_NUM_STREAMS;
                uint points_offset = i * points_elems;
                bool lastBatch = false;
                if (points_offset + points_elems > numPoints) {
                    lastBatch = true;
                    points_elems = numPoints - points_offset;
                }

                Vector3d *d_pointsData = nullptr;
                CUDA_CHECK(cudaMalloc((void **) &d_pointsData, sizeof(Vector3d) * points_elems));
                CUDA_CHECK(cudaMemcpyAsync(d_pointsData, pointsData.data() + points_offset, sizeof(Vector3d) * points_elems,
                                           cudaMemcpyHostToDevice, streams[i]));

                thrust::device_vector<double> d_value(points_elems); // b����ֵ

                if (useThrust)
                    mq_execMyReduce<Vector3d, double, true>(prop, streams[i], points_elems, numNodeVerts, paddingCols,
                                                       d_nodeVertexArray, d_nodeWidthArray, d_pointsData, d_lambda,
                                                       d_outerVal, d_minRange, d_maxRange, d_value);
                else
                    mq_execMyReduce<Vector3d, double, false>(prop, streams[i], points_elems, numNodeVerts, paddingCols,
                                                        d_nodeVertexArray, d_nodeWidthArray, d_pointsData, d_lambda,
                                                        d_outerVal, d_minRange, d_maxRange, d_value);

                CUDA_CHECK(cudaMemcpyAsync(h_bSplineVal + points_offset, d_value.data().get(),
                                           sizeof(double) * points_elems, cudaMemcpyDeviceToHost, streams[i]));

                CUDA_CHECK(cudaFree(d_pointsData));
                cleanupThrust(d_value);

                printf("[%d/%d] batch_size = %u", i + 1, MAX_NUM_STREAMS, points_elems);
                if (i != MAX_NUM_STREAMS - 1 && !lastBatch) printf("\r");
                else {
                    printf("\n");
                    break;
                }
            }

            for (int i = 0; i < MAX_NUM_STREAMS; i++)
                cudaStreamSynchronize(streams[i]);
            for (int i = 0; i < MAX_NUM_STREAMS; ++i)
                CUDA_CHECK(cudaStreamDestroy(streams[i]));

            bSplinVal = Eigen::Map<Eigen::VectorXd>(h_bSplineVal, numPoints);

            CUDA_CHECK(cudaFree(d_nodeVertexArray));
            CUDA_CHECK(cudaFree(d_nodeWidthArray));
            CUDA_CHECK(cudaFree(d_minRange));
            CUDA_CHECK(cudaFree(d_maxRange));
            CUDA_CHECK(cudaFree(d_lambda));
            CUDA_CHECK(cudaFree(d_outerVal));
            //free(h_bSplineVal);
        }

        /*__global__ void modelPointsMortonKernel(const uint nModelVerts,
            const Vector3d* d_modelOrigin, const double* d_nodeWidth,
            const Vector3d* d_modelVertsArray, uint32_t* d_vertsMorton)
        {
            const unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x;

            if (tx < nModelVerts)
            {
                const Vector3d modelOrigin = *d_modelOrigin;
                const Vector3d modelVert = d_modelVertsArray[tx];
                const double nodeWidth = *d_nodeWidth;

                const Vector3i dis = ((modelVert - modelOrigin).array() / nodeWidth).cast<int>();

                uint32_t mortonCode = morton::mortonEncode_LUT((uint16_t)dis.x(), (uint16_t)dis.y(), (uint16_t)dis.z());
                d_vertsMorton[tx] = mortonCode;
            }
        }

        void cpModelPointsMorton(const Vector3d& modelOrigin, const double& nodeWidth,
            const uint& nModelVerts, const vector<Vector3d> modelVertsArray, vector<uint32_t> vertsMorton)
        {
            vertsMorton.resize(nModelVerts, 0);
            thrust::device_vector<uint32_t> d_vertsMorton;
            thrust::device_vector<Vector3d> d_modelVertsArray = modelVertsArray;

            Vector3d* d_modelOrigin = nullptr;
            double* d_nodeWidth = nullptr;

            CUDA_CHECK(cudaMalloc((void**)&d_modelOrigin, sizeof(Vector3d)));
            CUDA_CHECK(cudaMemcpy(d_modelOrigin, &modelOrigin, sizeof(Vector3d), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc((void**)&d_nodeWidth, sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_nodeWidth, &nodeWidth, sizeof(double), cudaMemcpyHostToDevice));

            int minGridSize, gridSize, blockSize;
            getOccupancyMaxPotentialBlockSize(nModelVerts, minGridSize, gridSize, blockSize, modelPointsMortonKernel);

            modelPointsMortonKernel<<<gridSize, blockSize>>>(nModelVerts, d_modelOrigin, d_nodeWidth, d_modelVertsArray.data().get(), d_vertsMorton.data().get());

            CUDA_CHECK(cudaMemcpy(vertsMorton.data(), d_vertsMorton.data().get(), sizeof(uint32_t) * nModelVerts, cudaMemcpyDeviceToHost));

            cleanupThrust(d_vertsMorton);
            cleanupThrust(d_modelVertsArray);
            CUDA_CHECK(cudaFree(d_modelOrigin));
            CUDA_CHECK(cudaFree(d_nodeWidth));
        }*/
    }

NAMESPACE_END(ITS)