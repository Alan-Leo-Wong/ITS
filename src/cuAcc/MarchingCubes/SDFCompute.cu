#include "MCDefine.h"
#include "MarchingCubes.h"
#include "..\CUDACompute.h"
//#include "..\..\utils\cuda\cuBLASCheck.cuh"
#include "..\..\BSpline.hpp"
#include <device_launch_parameters.h>

/**
 * @brief 准备用于计算 sdf 的矩阵
 *
 * @param nVoxelElemCorners   等于每个 stream 下 voxel 的数目 nVoxelElems * 8，每个 stream 中 voxel 的顶点数
 * @param nAllNodeCorners     等于 nAllNodes * 8，所有节点的顶点数
 * @param d_voxelOffset       当前流中 voxel 相对于所有 voxel 的偏移量
 * @param d_res               分辨率
 * @param d_lambda            B样条系数
 * @param d_origin            MC算法被执行的初始区域原点坐标
 * @param d_voxelSize         每个 voxel 的大小
 * @param d_nodeCorners       每个八叉树节点的顶点
 * @param d_nodeWidth         每个八叉树节点的宽度
 * @param d_voxelMatrix       output matrix
 */
__global__ void MCKernel::prepareMatrixKernel(const uint nVoxelElemCorners,
	const uint nAllNodeCorners,
	const uint* d_voxelOffset,
	const uint3* d_res,
	double* d_lambda,
	double3* d_origin,
	double3* d_voxelSize,
	V3d* d_nodeCorners,
	V3d* d_nodeWidth,
	double* d_voxelMatrix)
{
	uint tx = blockIdx.x * blockDim.x + threadIdx.x;
	uint ty = blockIdx.y * blockDim.y + threadIdx.y;

	if (tx < nAllNodeCorners && ty < nVoxelElemCorners)
	{
		const uint voxel = ty / 8 + (*d_voxelOffset);

		uint3 voxelShift = getVoxelShift(voxel, *d_res);
		double3 origin = *d_origin;
		double3 voxelSize = *d_voxelSize;
		double3 voxelPos; // voxel 左下后角坐标

		voxelPos.x = origin.x + voxelShift.x * voxelSize.x;
		voxelPos.y = origin.y + voxelShift.y * voxelSize.y;
		voxelPos.z = origin.z + voxelShift.z * voxelSize.z;

		const uint voxelCorerOffset = ty % 8;
		double3 voxelCorner = voxelPos;

		if (voxelCorerOffset == 0) voxelCorner += make_double3(0, 0, 0);
		else if (voxelCorerOffset == 1) voxelCorner += make_double3(0, voxelSize.y, 0);
		else if (voxelCorerOffset == 2) voxelCorner += make_double3(voxelSize.x, voxelSize.y, 0);
		else if (voxelCorerOffset == 3) voxelCorner += make_double3(voxelSize.x, 0, 0);
		else if (voxelCorerOffset == 4) voxelCorner += make_double3(0, 0, voxelSize.z);
		else if (voxelCorerOffset == 5) voxelCorner += make_double3(0, voxelSize.y, voxelSize.z);
		else if (voxelCorerOffset == 6) voxelCorner += make_double3(voxelSize.x, voxelSize.y, voxelSize.z);
		else if (voxelCorerOffset == 7) voxelCorner += make_double3(voxelSize.x, 0, voxelSize.z);

		V3d width = d_nodeWidth[tx / 8];
		const uint idx = ty * nAllNodeCorners + tx;

		d_voxelMatrix[idx] = d_lambda[tx] * BaseFunction4Point(d_nodeCorners[tx], width, V3d(voxelCorner.x, voxelCorner.y, voxelCorner.z));
	}
}

void MC::launch_prepareMatrixKernel(const uint& nVoxelElems, const uint& nAllNodes, const uint& voxelOffset, const cudaStream_t& stream, double*& d_voxelMatrix)
{
	uint* d_voxelOffset = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&d_voxelOffset, sizeof(uint)));
	CUDA_CHECK(cudaMemcpyAsync(d_voxelOffset, &voxelOffset, sizeof(uint), cudaMemcpyHostToDevice, stream));

	// 计算矩阵(nVoxelElemCorners * 8 * nAllNodes * 8)
	const uint nVoxelElemCorners = nVoxelElems * 8;
	const uint nAllNodeCorners = nAllNodes * 8;
	uint voxelMatrixSize = nVoxelElemCorners * nAllNodeCorners;
	CUDA_CHECK(cudaMalloc((void**)&d_voxelMatrix, sizeof(double) * voxelMatrixSize));

	dim3 nThreads(P_NTHREADS_X, P_NTHREADS_Y, 1);
	assert(P_NTHREADS_X * P_NTHREADS_Y <= 1024, "P_NTHREADS_X * P_NTHREADS_Y is larger than 1024!\n");
	dim3 nBlocks((nAllNodeCorners + nThreads.x - 1) / nThreads.x, (nVoxelElemCorners + nThreads.y - 1) / nThreads.y, 1);

	MCKernel::prepareMatrixKernel << <nBlocks, nThreads >> > (nVoxelElemCorners, nAllNodeCorners,
		d_voxelOffset, d_res, d_lambda, d_gridOrigin,
		d_voxelSize, d_nodeCorners, d_nodeWidth, d_voxelMatrix);
	getLastCudaError("Kernel: 'prepareMatrixKernel' failed!\n");

}

void MC::launch_computSDFKernel(const uint& nVoxels)
{
	cudaStream_t streams[MAX_NUM_STREAMS];
	for (int i = 0; i < MAX_NUM_STREAMS; ++i)
		CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

	for (int i = 0; i < MAX_NUM_STREAMS; ++i)
	{
		// nVoxelElems: the number of voxels for each stream
		uint nVoxelElems = (nVoxels + MAX_NUM_STREAMS - 1) / MAX_NUM_STREAMS;
		uint voxelOffset = i * nVoxelElems;
		nVoxelElems = voxelOffset + nVoxelElems > nVoxels ? nVoxels - voxelOffset : nVoxelElems;

		double* d_voxelMatrix = nullptr;

		launch_prepareMatrixKernel(nVoxelElems, nAllNodes, voxelOffset, streams[i], d_voxelMatrix);

		launch_BLASRowSumReduce(streams[i], nVoxelElems * 8, nAllNodes * 8, d_voxelMatrix, d_voxelSDF + voxelOffset * 8);

		CUDA_CHECK(cudaFree(d_voxelMatrix));
	}

	for (int i = 0; i < MAX_NUM_STREAMS; ++i)
		CUDA_CHECK(cudaStreamDestroy(streams[i]));
}