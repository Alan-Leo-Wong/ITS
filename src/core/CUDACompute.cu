#include "CUDACompute.hpp"
#include "MortonLUT.hpp"
#include "BSpline.hpp"
#include "utils/Common.hpp"
#include "utils/cuda/CUDAUtil.cuh"
#include "utils/cuda/DeviceQuery.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/generate.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

NAMESPACE_BEGIN(ITS)
    namespace cuAcc {
        namespace cg = cooperative_groups;
        using namespace utils::cuda;

        namespace detail {
            /**
             * Convert a linear index to a row index. Use for sum reduction in thrust.
             * @tparam T
             */
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

            template<typename Real>
            __global__ void modelTriAttributeKernel(size_t nTriangles,
                                                    Triangle <Real> *d_modelTriangleArray) {
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

            /**
             * Sum reduction at warp level.
             * @tparam T
             * @param mask
             * @param sum
             */
            template<typename T>
            __device__ __forceinline__ void warpReduceSum(unsigned int mask, T &sum) {
                for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
                    sum += __shfl_down_sync(mask, sum, offset);
            }

            /**
             * My matrix row sum reduction kernel.
             * @tparam T
             * @tparam Scalar
             * @tparam nIsPow2
             * @tparam colBlockSize
             * @param m rows
             * @param n columns, and n % warpSize = 0
             * @param d_nodeVertexArray
             * @param d_nodeWidthArray
             * @param g_iA
             * @param g_iB
             * @param g_odata
             */
            template<typename T = Eigen::Vector3d, typename Scalar = double, bool nIsPow2, unsigned int colBlockSize>
            __global__ void reduceRowSumKernel(unsigned int m, unsigned int n,
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
                            sum += g_iB[i] * core::bSplineForPoint(d_nodeVertexArray[i].first,
                                                                   d_nodeWidthArray[d_nodeVertexArray[i].second],
                                                                   g_iA[ty]);
                            if (i + colBlockSize < n) {
                                sum += g_iB[i + colBlockSize] *
                                       core::bSplineForPoint(d_nodeVertexArray[i + colBlockSize].first,
                                                             d_nodeWidthArray[d_nodeVertexArray[i +
                                                                                                colBlockSize].second],
                                                             g_iA[ty]); // (һ��)�߳̿鼶��Ŀ��
                                i += x_gridSize; // ���񼶱�Ŀ�ȣ�Ĭ�������С(block������)Ϊԭ������(xά�ȼ�����)��һ��(���nIsPow2��������x_gridSize����һ��)
                            }
                        }
                    } else {
                        unsigned int i = blockIdx.x * colBlockSize + threadIdx.x;
                        while (i < n) {
                            //printf("#2 sum = %lf\n", sum);
                            sum += g_iB[i] * core::bSplineForPoint(d_nodeVertexArray[i].first,
                                                                   d_nodeWidthArray[d_nodeVertexArray[i].second],
                                                                   g_iA[ty]);
                            i += x_gridSize;
                        }
                    }

                    // perform reduction summation for each warp,
                    // then save the result to shared memory.
                    warpReduceSum<Scalar>(mask, sum);
                    const int sh_reduceNum = (colBlockSize / warpSize) > 0 ? colBlockSize / warpSize : 1;
                    if (x_tid % warpSize == 0)
                        shData[threadIdx.y * sh_reduceNum + x_tid / warpSize] = sum;

                    cg::sync(ctb);

                    //printf("#3 sum = %lf\n", sum);

                    // Summation of all warps within the same block (only adding the sum saved by the first thread of each warp,
                    // because the sum saved by the first thread of each warp is the sum of data from all threads in its warp)
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

            /**
             * My matrix row sum reduction kernel. Use for multi points query.
             * @tparam T
             * @tparam Scalar
             * @tparam nIsPow2
             * @tparam colBlockSize
             * @param m
             * @param n
             * @param d_nodeVertexArray
             * @param d_nodeWidthArray
             * @param g_iA
             * @param g_iB
             * @param d_outerVal
             * @param d_minRange
             * @param d_maxRange
             * @param g_odata
             */
            template<typename T = Eigen::Vector3d, typename Scalar = double, bool nIsPow2, unsigned int colBlockSize>
            __global__ void mq_reduceRowSumKernel(unsigned int m, unsigned int n,
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
                            sum += g_iB[i] * core::bSplineForPoint(d_nodeVertexArray[i].first,
                                                                   d_nodeWidthArray[d_nodeVertexArray[i].second],
                                                                   g_iA[ty]);
                            if (i + colBlockSize < n) {
                                sum += g_iB[i + colBlockSize] *
                                       core::bSplineForPoint(d_nodeVertexArray[i + colBlockSize].first,
                                                             d_nodeWidthArray[d_nodeVertexArray[i +
                                                                                                colBlockSize].second],
                                                             g_iA[ty]); // (һ��)�߳̿鼶��Ŀ��
                                i += x_gridSize; // ���񼶱�Ŀ�ȣ�Ĭ�������С(block������)Ϊԭ������(xά�ȼ�����)��һ��(���nIsPow2��������x_gridSize����һ��)
                            }
                        }
                    } else {
                        unsigned int i = blockIdx.x * colBlockSize + threadIdx.x;
                        while (i < n) {
                            sum += g_iB[i] * core::bSplineForPoint(d_nodeVertexArray[i].first,
                                                                   d_nodeWidthArray[d_nodeVertexArray[i].second],
                                                                   g_iA[ty]);
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

            /**
             * Switch kernel for my row sum reduction.
             * @tparam T
             * @tparam Scalar
             * @param isPow2
             * @param threads
             * @param gridSize
             * @param blockSize
             * @param sh_memSize
             * @param stream
             * @param rowElems
             * @param cols
             * @param d_nodeVertexArray
             * @param d_nodeWidthArray
             * @param d_A
             * @param d_B
             * @param d_tRowSumMatrix
             */
            template<typename T = Eigen::Vector3d, typename Scalar = double>
            void switchKernel(bool isPow2, int threads, const dim3 &gridSize,
                              const dim3 &blockSize, int sh_memSize, const cudaStream_t &stream,
                              uint rowElems,
                              uint cols, const thrust::pair<T, uint32_t> *d_nodeVertexArray,
                              const T *d_nodeWidthArray, const T *d_A, const Scalar *d_B, Scalar *d_tRowSumMatrix) {
                using namespace detail;
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

            /**
             * Switch kernel for my row sum reduction. Use for multi points query.
             * @tparam T
             * @tparam Scalar
             * @param isPow2
             * @param threads
             * @param gridSize
             * @param blockSize
             * @param sh_memSize
             * @param stream
             * @param rowElems
             * @param cols
             * @param d_nodeVertexArray
             * @param d_nodeWidthArray
             * @param d_A
             * @param d_B
             * @param d_outerVal
             * @param d_minRange
             * @param d_maxRange
             * @param d_tRowSumMatrix
             */
            template<typename T = Eigen::Vector3d, typename Scalar = double>
            void mq_switchKernel(bool isPow2, int threads, const dim3 &gridSize,
                                 const dim3 &blockSize, int sh_memSize, const cudaStream_t &stream,
                                 uint rowElems,
                                 uint cols, const thrust::pair<T, uint32_t> *d_nodeVertexArray,
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

            /**
             * My matrix sum reduction of each row.
             *
             * Partition A and B into blocks, with each block computing a partial sum.
             * Specifically, A is divided into rowElems blocks along the row direction,
             * and B is divided into n (x_gridSize) blocks along the column direction.
             * Each row block has a size of x_blockSize,
             * and each column block has a size of y_blockSize.
             * For each row block, compute the corresponding
             * \sum func(ai, bj) for (i:1->x_blockSize, j:1->y_blockSize).
             * Finally, obtain d_tRowSumMatrix(rowElems, n).
             * @tparam T
             * @tparam Scalar
             * @tparam useThrust
             * @param prop
             * @param stream
             * @param rowElems
             * @param cols
             * @param paddingCols
             * @param d_nodeVertexArray
             * @param d_nodeWidthArray
             * @param d_A
             * @param d_B
             * @param d_value
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

                // Padding columns is needed during allocation.
                getBlocksAndThreadsNum(prop, paddingCols, 65535, 1024 / y_blockSize, x_gridSize, x_blockSize);
                dim3 blockSize(x_blockSize, y_blockSize, 1);
                dim3 gridSize(x_gridSize, y_gridSize, 1);

                unsigned int x_paddingGridSize = PADDING_TO_WARP(x_gridSize);
                unsigned int t_rowSumMatrixSize =
                        rowElems * x_paddingGridSize; // ����ʱ��Ҫpadding���cols��������Ϊ�����ں����ظ�����row reduce sum

                thrust::device_vector<Scalar> d_tRowSumMatrix(t_rowSumMatrixSize, (Scalar) .0);
                int sh_memSize =
                        sizeof(Scalar) * y_blockSize * ((x_blockSize / 32) + 1); // +1 for avoiding bank conflicts
                bool flag = isPow2(cols);

                // d_tRowSumMatrix Ϊ row reduce sum �Ľ������ʵ�ʲ��� 0 ������ά��Ϊ: elems * x_gridSize��
                // ������ elems * x_paddingGridSize
                switchKernel<T, Scalar>(flag, x_blockSize, gridSize, blockSize, sh_memSize, stream, rowElems,
                                        cols, d_nodeVertexArray, d_nodeWidthArray, d_A, d_B,
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

            /**
             * My matrix sum reduction of each row. Use for multi points query.
             * @tparam T
             * @tparam Scalar
             * @tparam useThrust
             * @param prop
             * @param stream
             * @param rowElems
             * @param cols
             * @param paddingCols
             * @param d_nodeVertexArray
             * @param d_nodeWidthArray
             * @param d_A
             * @param d_B
             * @param d_outerVal
             * @param d_minRange
             * @param d_maxRange
             * @param d_value
             */
            template<typename T = Eigen::Vector3d, typename Scalar = double, bool useThrust = true>
            void mq_execMyReduce(const cudaDeviceProp &prop, const cudaStream_t &stream,
                                 uint rowElems, uint cols, uint paddingCols,
                                 const thrust::pair<T, uint32_t> *d_nodeVertexArray,
                                 const T *d_nodeWidthArray, const T *d_A, const Scalar *d_B, const Scalar *d_outerVal,
                                 const Eigen::Array3d *d_minRange, const Eigen::Array3d *d_maxRange,
                                 thrust::device_vector<Scalar> &d_value) {
                int x_blockSize = 0, y_blockSize = 32; // x����B��y����A (ע�⣺x_blockSize �����>=32)
                int x_gridSize = 0, y_gridSize = (rowElems + y_blockSize - 1) / y_blockSize;

                // Padding columns is needed during allocation.
                getBlocksAndThreadsNum(prop, paddingCols, 65535, 1024 / y_blockSize, x_gridSize, x_blockSize);
                dim3 blockSize(x_blockSize, y_blockSize, 1);
                dim3 gridSize(x_gridSize, y_gridSize, 1);

                unsigned int x_paddingGridSize = PADDING_TO_WARP(x_gridSize);
                unsigned int t_rowSumMatrixSize =
                        rowElems * x_paddingGridSize; // ����ʱ��Ҫpadding���cols��������Ϊ�����ں����ظ�����row reduce sum

                thrust::device_vector<Scalar> d_tRowSumMatrix(t_rowSumMatrixSize, (Scalar) .0);
                int sh_memSize =
                        sizeof(Scalar) * y_blockSize * ((x_blockSize / 32) + 1); // +1 for avoiding bank conflicts
                bool flag = isPow2(cols);

                // 'd_tRowSumMatrix' is the result of row reduction sum,
                // its actual non-zero data dimension is: 'elems * x_gridSize', not 'elems * x_paddingGridSize'.
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
        } // namespace detail

        void launch_modelTriAttributeKernel(size_t nTriangles,
                                            std::vector<Triangle<Eigen::Vector3d>> &modelTriangleArray) {
            thrust::device_vector<Triangle<Eigen::Vector3d>> d_modelTriangleArray = modelTriangleArray;
            int blockSize, gridSize, minGridSize;
            getOccupancyMaxPotentialBlockSize(nTriangles, minGridSize, blockSize, gridSize,
                                              detail::modelTriAttributeKernel<Eigen::Vector3d>, 0, 0);
            detail::modelTriAttributeKernel<Eigen::Vector3d> << <
            gridSize, blockSize >> > (nTriangles, d_modelTriangleArray.data().get());
            getLastCudaError("Kernel 'modelTriAttributeKernel' launch failed!\n");

            CUDA_CHECK(cudaMemcpy(modelTriangleArray.data(), d_modelTriangleArray.data().get(),
                                  sizeof(Triangle<Eigen::Vector3d>) * nTriangles, cudaMemcpyDeviceToHost));
        }

        void launch_BLASRowSumReduce(int rows,
                                     int columns,
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

        void launch_ThrustRowSumReduce(int rows,
                                       int columns,
                                       const thrust::device_vector<double> &d_matrix,
                                       thrust::device_vector<double> &row_sums,
                                       const cudaStream_t &stream) {
            if (row_sums.size() != rows) {
                row_sums.clear();
                row_sums.resize(rows);
            }
            // assert(row_sums.size() == rows, "THE SIZE OF ROW_SUMS IS NOT EQUAL ROWS!\n");

            thrust::device_vector<int> row_indices(rows);

            if (stream)
                thrust::reduce_by_key(thrust::cuda::par.on(stream),
                                      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                                      detail::linear_index_to_row_index<int>(columns)),
                                      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                                      detail::linear_index_to_row_index<int>(rows)) +
                                      (rows * columns),
                                      d_matrix.begin(),
                                      row_indices.begin(),
                                      row_sums.begin(),
                                      thrust::equal_to<int>(),
                                      thrust::plus<double>());
            else
                thrust::reduce_by_key(thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                                      detail::linear_index_to_row_index<int>(columns)),
                                      thrust::make_transform_iterator(thrust::counting_iterator<int>(0),
                                                                      detail::linear_index_to_row_index<int>(columns)) +
                                      (rows * columns),
                                      d_matrix.begin(),
                                      row_indices.begin(),
                                      row_sums.begin(),
                                      thrust::equal_to<int>(),
                                      thrust::plus<double>());
        }

        void cpBSplineVal(uint numNodeVerts, uint numNodes,
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
                detail::execMyReduce<Vector3d, double, true>(prop, 0, 1, numNodeVerts, paddingCols, d_nodeVertexArray,
                                                             d_nodeWidthArray, d_pointData, d_lambda, d_value);
            else
                detail::execMyReduce<Vector3d, double, false>(prop, 0, 1, numNodeVerts, paddingCols, d_nodeVertexArray,
                                                              d_nodeWidthArray, d_pointData, d_lambda, d_value);

            CUDA_CHECK(cudaMemcpy(&bSplinVal, d_value.data().get(), sizeof(double), cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaFree(d_nodeVertexArray));
            CUDA_CHECK(cudaFree(d_nodeWidthArray));
            CUDA_CHECK(cudaFree(d_lambda));
            CUDA_CHECK(cudaFree(d_pointData));
            //free(h_bSplineVal);
        }

        void cpBSplineVal(uint numPoints, uint numNodeVerts, uint numNodes,
                          const std::vector<Vector3d> &pointsData,
                          const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>> &nodeVertexArray,
                          const std::vector<Vector3d> &nodeWidthArray, const VectorXd &lambda, VectorXd &bSplinVal,
                          bool useThrust) {
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
                CUDA_CHECK(cudaMemcpyAsync(d_pointsData, pointsData.data() + points_offset,
                                           sizeof(Vector3d) * points_elems,
                                           cudaMemcpyHostToDevice, streams[i]));

                thrust::device_vector<double> d_value(points_elems);

                if (useThrust)
                    detail::execMyReduce<Vector3d, double, true>(prop, streams[i], points_elems, numNodeVerts,
                                                                 paddingCols,
                                                                 d_nodeVertexArray, d_nodeWidthArray, d_pointsData,
                                                                 d_lambda,
                                                                 d_value);
                else
                    detail::execMyReduce<Vector3d, double, false>(prop, streams[i], points_elems, numNodeVerts,
                                                                  paddingCols,
                                                                  d_nodeVertexArray, d_nodeWidthArray, d_pointsData,
                                                                  d_lambda,
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

        void cpBSplineVal(const cudaDeviceProp &prop, uint numPoints, uint numNodeVerts,
                          const thrust::pair<Eigen::Vector3d, uint32_t> *d_nodeVertexArray,
                          const Vector3d *d_nodeWidthArray,
                          const double *d_lambda, const thrust::device_vector<Vector3d> &d_pointsData,
                          thrust::device_vector<double> &d_bSplineVal, const cudaStream_t &stream,
                          bool useThrust) {
            unsigned int paddingCols = PADDING_TO_WARP(numNodeVerts);

            if (useThrust)
                detail::execMyReduce<Vector3d, double, true>(prop, stream, numPoints, numNodeVerts, paddingCols,
                                                             d_nodeVertexArray,
                                                             d_nodeWidthArray, d_pointsData.data().get(), d_lambda,
                                                             d_bSplineVal);
            else
                detail::execMyReduce<Vector3d, double, false>(prop, stream, numPoints, numNodeVerts, paddingCols,
                                                              d_nodeVertexArray,
                                                              d_nodeWidthArray, d_pointsData.data().get(), d_lambda,
                                                              d_bSplineVal);
        }

        void cpPointQuery(uint numPoints, uint numNodeVerts,
                          uint numNodes, const Array3d &minRange, const Array3d &maxRange,
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
                CUDA_CHECK(cudaMemcpyAsync(d_pointsData, pointsData.data() + points_offset,
                                           sizeof(Vector3d) * points_elems,
                                           cudaMemcpyHostToDevice, streams[i]));

                thrust::device_vector<double> d_value(points_elems); // b����ֵ

                if (useThrust)
                    detail::mq_execMyReduce<Vector3d, double, true>(prop, streams[i], points_elems, numNodeVerts,
                                                                    paddingCols,
                                                                    d_nodeVertexArray, d_nodeWidthArray, d_pointsData,
                                                                    d_lambda,
                                                                    d_outerVal, d_minRange, d_maxRange, d_value);
                else
                    detail::mq_execMyReduce<Vector3d, double, false>(prop, streams[i], points_elems, numNodeVerts,
                                                                     paddingCols,
                                                                     d_nodeVertexArray, d_nodeWidthArray, d_pointsData,
                                                                     d_lambda,
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

    } // namespace cuAcc
NAMESPACE_END(ITS)