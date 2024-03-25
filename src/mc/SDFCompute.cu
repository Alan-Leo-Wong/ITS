#include "MarchingCubes.h"
#include "MCDefine.hpp"
#include "core/CUDACompute.hpp"
//#include "..\..\utils\cuda\cuBLASCheck.cuh"
#include "core/BSpline.hpp"
#include <device_launch_parameters.h>

/**
 * @brief ׼�����ڼ��� sdf �ľ���
 *
 * @param nVoxelElemCorners   ����ÿ�� stream �� voxel ����Ŀ nVoxelElems * 8��ÿ�� stream �� voxel �Ķ�����
 * @param nAllNodeCorners     ���� nAllNodes * 8�����нڵ�Ķ�����
 * @param d_voxelOffset       ��ǰ���� voxel ��������� voxel ��ƫ����
 * @param d_res               �ֱ���
 * @param d_lambda            B����ϵ��
 * @param d_origin            MC�㷨��ִ�еĳ�ʼ����ԭ������
 * @param d_voxelSize         ÿ�� voxel �Ĵ�С
 * @param d_nodeCorners       ÿ���˲����ڵ�Ķ���
 * @param d_nodeWidth         ÿ���˲����ڵ�Ŀ��
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
		double3 voxelPos; // voxel ���º������

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

	// �������(nVoxelElemCorners * 8 * nAllNodes * 8)
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