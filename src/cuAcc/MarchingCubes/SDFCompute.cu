#include "MCDefine.h"
#include "MarchingCubes.h"
#include "..\CUDACompute.h"
#include "..\..\BSpline.hpp"
#include <device_launch_parameters.h>

/**
 * @brief ׼�����ڼ��� sdf �ľ���
 *
 * @param nVoxelElems   ����ÿ�� stream �� voxel ����Ŀ nVoxelElems * 8��ÿ�� stream �� voxel �Ķ�����
 * @param nAllNodes     ���� nAllNodes * 8�����нڵ�Ķ�����
 * @param d_voxelOffset ��ǰ���� voxel ��������� voxel ��ƫ����
 * @param d_res         �ֱ���
 * @param d_lambda      B����ϵ��
 * @param d_origin      MC�㷨��ִ�еĳ�ʼ����ԭ������
 * @param d_voxelSize   ÿ�� voxel �Ĵ�С
 * @param d_nodeCorners ÿ���˲����ڵ�Ķ���
 * @param d_nodeWidth   ÿ���˲����ڵ�Ŀ��
 * @param d_voxelMatrix output matrix
 */
__global__ void MCKernel::prepareMatrixKernel(const uint nVoxelElems,
	const uint nAllNodes,
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

	if (tx < nAllNodes && ty < nVoxelElems)
	{
		const uint voxel = ty + (*d_voxelOffset);

		uint3 voxelShift = getVoxelShift(voxel, *d_res);
		double3 origin = *d_origin;
		double3 voxelSize = *d_voxelSize;
		double3 voxelPos; // voxel ���º������

		voxelPos.x = origin.x + voxelShift.x * voxelSize.x;
		voxelPos.y = origin.y + voxelShift.y * voxelSize.y;
		voxelPos.z = origin.z + voxelShift.z * voxelSize.z;

		// corners of current voxel
		double3 corners[8];
		corners[0] = voxelPos;
		corners[1] = voxelPos + make_double3(0, voxelSize.y, 0);
		corners[2] = voxelPos + make_double3(voxelSize.x, voxelSize.y, 0);
		corners[3] = voxelPos + make_double3(voxelSize.x, 0, 0);
		corners[4] = voxelPos + make_double3(0, 0, voxelSize.z);
		corners[5] = voxelPos + make_double3(0, voxelSize.y, voxelSize.z);
		corners[6] = voxelPos + make_double3(voxelSize.x, voxelSize.y, voxelSize.z);
		corners[7] = voxelPos + make_double3(voxelSize.x, 0, voxelSize.z);

		V3d width = d_nodeWidth[tx];

		for (int k = 0; k < 8; ++k)
		{
			const int nodeCornerIdx = tx * 8 + k;
			double3 corner = corners[nodeCornerIdx];
			const int idx = (voxel + k) * nAllNodes * 8 + nodeCornerIdx;

			d_voxelMatrix[idx] = d_lambda[nodeCornerIdx] * BaseFunction4Point(d_nodeCorners[nodeCornerIdx], width, V3d(corner.x, corner.y, corner.z));
		}
	}
}

void MC::launch_prepareMatrixKernel(const uint& nVoxelElems, const uint& voxelOffset, const cudaStream_t& stream, double* d_voxelMatrix)
{
	uint* d_voxelOffset = nullptr;
	CUDA_CHECK(cudaMalloc((void**)&d_voxelOffset, sizeof(uint)));
	CUDA_CHECK(cudaMemcpyAsync(d_voxelOffset, &voxelOffset, sizeof(uint), cudaMemcpyHostToDevice, stream));

	// �������(nVoxelElemCorners * 8 * nAllNodes * 8)
	const uint nVoxelElemCorners = nVoxelElems * 8;
	const uint nAllNodeCorners = nAllNodes * 8;
	uint voxelMatrixSize = nVoxelElemCorners * nAllNodeCorners;
	CUDA_CHECK(cudaMalloc((void**)&d_voxelMatrix, sizeof(double) * voxelMatrixSize));

	dim3 nThreads(P_NTHREADS_X, P_NTHREADS_Y, 1);
	assert(P_NTHREADS_X * P_NTHREADS_Y <= 1024, "P_NTHREADS_X * P_NTHREADS_Y is larger than 1024!\n");
	dim3 nBlocks((nAllNodes + nThreads.x - 1) / nThreads.x, (nVoxelElems + nThreads.y - 1) / nThreads.y, 1);

	MCKernel::prepareMatrixKernel << <nBlocks, nThreads >> > (nVoxelElems, nAllNodes,
		d_voxelOffset, d_res, d_lambda, d_gridOrigin,
		d_voxelSize, d_nodeCorners, d_nodeWidth, d_voxelMatrix);

	CUDA_CHECK(cudaFree(d_voxelOffset));
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

		launch_prepareMatrixKernel(nVoxelElems, voxelOffset, streams[i], d_voxelMatrix);

		launch_BLASRowSumReduce(streams[i], nVoxelElems * 8, nAllNodes * 8, d_voxelMatrix, d_voxelSDF + voxelOffset * 8);

		CUDA_CHECK(cudaFree(d_voxelMatrix));
	}

	for (int i = 0; i < MAX_NUM_STREAMS; i++)
		cudaStreamSynchronize(streams[i]);
	for (int i = 0; i < MAX_NUM_STREAMS; ++i)
		CUDA_CHECK(cudaStreamDestroy(streams[i]));
}