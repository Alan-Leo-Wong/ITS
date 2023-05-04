#include "MCDefine.h"
#include "LookTable.h"
#include "MarchingCubes.h"
#include "..\..\utils\String.hpp"
#include "..\..\BSpline.hpp"
#include "..\..\utils\cuda\CUDAUtil.cuh"
#include "..\..\cuAcc\CUDACompute.h"
#include "..\..\utils\cuda\DeviceQuery.cuh"
#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <fstream>
#include <texture_types.h>
#include <thrust\device_vector.h>
#include <thrust\scan.h>
#include <vector_functions.h>
#include <vector_types.h>

namespace MC {
	// host
	//namespace {
	uint numNodeVerts = 0;

	uint allTriVertices = 0, nValidVoxels = 0;

	double3* h_triPoints = nullptr; // output
	//} // namespace

	// device
	//namespace {
	thrust::device_vector<SVONode> d_svoNodeArray;
	thrust::pair<Eigen::Vector3d, uint32_t>* d_nodeVertexArray;

	double* d_lambda = nullptr;
	V3d* d_nodeWidthArray = nullptr;

	uint3* d_res = nullptr;
	double* d_isoVal = nullptr;

	uint* d_nVoxelVertsArray = nullptr;
	uint* d_nVoxelVertsScan = nullptr;

	uint* d_isValidVoxelArray = nullptr;
	uint* d_nValidVoxelsScan = nullptr;

	double3* d_gridOrigin = nullptr;
	double3* d_voxelSize = nullptr;

	thrust::device_vector<double> d_voxelSDF;
	static thrust::host_vector<double> h_voxelSDF;

	//static double* d_voxelSDF = nullptr;
	uint* d_voxelCubeIndex = nullptr;

	uint* d_compactedVoxelArray = nullptr;

	int* d_triTable = nullptr;
	int* d_nVertsTable = nullptr;

	// textures containing look-up tables
	cudaTextureObject_t triTex;
	cudaTextureObject_t nVertsTex;

	double3* d_triPoints = nullptr; // output
	//} // namespace
} // namespace MC

__device__ inline double3 MCKernel::vertexLerp(const double3 p_0,
	const double3 p_1,
	const double sdf_0,
	const double sdf_1,
	const double isoVal) {
	if (abs(isoVal - sdf_0) < 1e-6)
		return p_0;
	if (abs(isoVal - sdf_1) < 1e-6)
		return p_1;
	if (abs(sdf_1 - sdf_0) < 1e-6)
		return p_0;

	double t = (isoVal - sdf_0) / (sdf_1 - sdf_0);
	double3 lerp_p;
	lerp_p.x = lerp(p_0.x, p_1.x, t);
	lerp_p.y = lerp(p_0.y, p_1.y, t);
	lerp_p.z = lerp(p_0.z, p_1.z, t);
	return lerp_p;
}

__device__ uint3 MCKernel::getVoxelShift(const uint index,
	const uint3 d_res) {
	uint x = index % d_res.x;
	uint y = index % (d_res.x * d_res.y) / d_res.x;
	uint z = index / (d_res.x * d_res.y);
	return make_uint3(x, y, z);
}

//__device__ inline double MCKernel::computeSDF(const uint nAllNodes, const V3d* d_nodeCorners,
//	const V3d* d_nodeWidth, const double* d_lambda, double3 pos)
//{
//	double sum = 0.0;
//	for (int i = 0; i < nAllNodes; ++i)
//	{
//		V3d width = d_nodeWidth[i];
//		for (int j = 0; j < 8; ++j)
//			sum += d_lambda[i * 8 + j] * BaseFunction4Point(d_nodeCorners[i * 8 + j], width, V3d(pos.x, pos.y, pos.z));
//	}
//	return sum;
//}
__device__ inline double MCKernel::computeSDF(const uint numNodeVerts, const thrust::pair<Eigen::Vector3d, uint32_t>* d_nodeVertexArray,
	const SVONode* d_svoNodeArray, const double* d_lambda, double3 pos)
{
	double sum = 0.0;
	for (int i = 0; i < numNodeVerts; ++i)
	{
		double width = d_svoNodeArray[d_nodeVertexArray[i].second].width;
		sum += d_lambda[i] * BaseFunction4Point(d_nodeVertexArray[i].first, width, V3d(pos.x, pos.y, pos.z));
	}
	return sum;
}

__global__ void MCKernel::prepareVoxelCornerKernel(const uint nVoxels, const uint* d_voxelOffset,
	const double3* d_origin, const double3* d_voxelSize, const uint3* d_res, V3d* d_voxelCornerData)
{
	const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < nVoxels)
	{
		const size_t voxelIdx = tid + (*d_voxelOffset);
		uint3 voxelShift = getVoxelShift(voxelIdx, *d_res);

		double3 origin = *d_origin;
		double3 voxelSize = *d_voxelSize;
		V3d voxelPos; // voxel 原点坐标

		voxelPos.x() = origin.x + voxelShift.x * voxelSize.x;
		voxelPos.y() = origin.y + voxelShift.y * voxelSize.y;
		voxelPos.z() = origin.z + voxelShift.z * voxelSize.z;

		d_voxelCornerData[tid * 8] = voxelPos;
		d_voxelCornerData[tid * 8 + 1] = voxelPos + V3d(0, voxelSize.y, 0);
		d_voxelCornerData[tid * 8 + 2] = voxelPos + V3d(voxelSize.x, voxelSize.y, 0);
		d_voxelCornerData[tid * 8 + 3] = voxelPos + V3d(voxelSize.x, 0, 0);
		d_voxelCornerData[tid * 8 + 4] = voxelPos + V3d(0, 0, voxelSize.z);
		d_voxelCornerData[tid * 8 + 5] = voxelPos + V3d(0, voxelSize.y, voxelSize.z);
		d_voxelCornerData[tid * 8 + 6] = voxelPos + V3d(voxelSize.x, voxelSize.y, voxelSize.z);
		d_voxelCornerData[tid * 8 + 7] = voxelPos + V3d(voxelSize.x, 0, voxelSize.z);
	}
}

/**
 * @brief 通过每个体素的sdf值确定分布情况
 *
 * @param nVoxels             voxel的总数量 = res_x * res_y * res_z
 * @param d_isoVal            isosurface value
 * @param nVertsTex           存储lookTabel对应数据的纹理对象
 * @param d_nVoxelVerts       经过 cube index 映射后每个 voxel 内应含点的数量
 * @param d_VoxelCubeIndex    每个 voxel 内 sdf 分布所对应的 cube index
 * @param d_voxelSDF          每个 voxel 八个顶点的 sdf 值
 * @param d_isValidVoxelArray 判断每个 voxel 是否是合理的 voxel
 */
__global__ void MCKernel::determineVoxelKernel(const uint nVoxels,
	const double* d_isoVal,
	const cudaTextureObject_t nVertsTex,
	uint* d_nVoxelVerts,
	uint* d_voxelCubeIndex,
	double* d_voxelSDF,
	uint* d_isValidVoxelArray) {
	uint bid = blockIdx.y * gridDim.x + blockIdx.x;
	uint tid = bid * blockDim.x + threadIdx.x;

	if (tid < nVoxels) {
		double isoVal = *d_isoVal;

		double sdf[8];

#pragma unroll
		for (int i = 0; i < 8; ++i)
			sdf[i] = d_voxelSDF[tid * 8 + i];

		int cubeIndex = 0;
		cubeIndex = (uint(sdf[0] < isoVal)) | (uint(sdf[1] < isoVal) << 1) |
			(uint(sdf[2] < isoVal) << 2) | (uint(sdf[3] < isoVal) << 3) |
			(uint(sdf[4] < isoVal) << 4) | (uint(sdf[5] < isoVal) << 5) |
			(uint(sdf[6] < isoVal) << 6) | (uint(sdf[7] < isoVal) << 7);

		int nVerts = tex1Dfetch<int>(nVertsTex, cubeIndex);
		d_nVoxelVerts[tid] = nVerts;
		d_isValidVoxelArray[tid] = nVerts > 0;
		d_voxelCubeIndex[tid] = cubeIndex;
	}
}

/**
 * @brief 计算每个体素的sdf值以及确定分布情况
 *
 * @param nVoxels             voxel的总数量 = res_x * res_y * res_z
 * @param d_isoVal            isosurface value
 * @param d_origin            MC算法被执行的初始区域原点坐标
 * @param d_voxelSize         每个 voxel 的大小
 * @param d_res               分辨率
 * @param d_nodeCorners       格子顶点
 * @param d_nodeWidth         格子宽度
 * @param d_lambda            格子宽度
 * @param nVertsTex           存储lookTabel对应数据的纹理对象
 * @param d_nVoxelVertsArray  经过 cube index 映射后每个 voxel 内应含点的数量的数组
 * @param d_VoxelCubeIndex    每个 voxel 内 sdf 分布所对应的 cube index
 * @param d_voxelSDF          每个 voxel 八个顶点的 sdf 值
 * @param d_isValidVoxelArray 判断每个 voxel 是否是合理的 voxel
 */
__global__ void MCKernel::determineVoxelKernel_2(const uint nVoxels, const uint numNodeVerts, const double* d_isoVal,
	const double3* d_origin, const double3* d_voxelSize, const uint3* d_res, const thrust::pair<Eigen::Vector3d, uint32_t>* d_nodeVertexArray,
	const SVONode* d_svoNodeArray, const double* d_lambda, const cudaTextureObject_t nVertsTex, uint* d_nVoxelVertsArray,
	uint* d_voxelCubeIndex, double* d_voxelSDF, uint* d_isValidVoxelArray)
{
	uint bid = blockIdx.y * gridDim.x + blockIdx.x;
	uint tid = bid * blockDim.x + threadIdx.x;

	if (tid < nVoxels) {
		double isoVal = *d_isoVal;

		uint3 voxelShift = getVoxelShift(tid, *d_res);
		double3 origin = *d_origin;
		double3 voxelSize = *d_voxelSize;
		double3 voxelPos; // voxel 原点坐标

		voxelPos.x = origin.x + voxelShift.x * voxelSize.x;
		voxelPos.y = origin.y + voxelShift.y * voxelSize.y;
		voxelPos.z = origin.z + voxelShift.z * voxelSize.z;

		double3 corners[8];
		corners[0] = voxelPos;
		corners[1] = voxelPos + make_double3(0, voxelSize.y, 0);
		corners[2] = voxelPos + make_double3(voxelSize.x, voxelSize.y, 0);
		corners[3] = voxelPos + make_double3(voxelSize.x, 0, 0);
		corners[4] = voxelPos + make_double3(0, 0, voxelSize.z);
		corners[5] = voxelPos + make_double3(0, voxelSize.y, voxelSize.z);
		corners[6] = voxelPos + make_double3(voxelSize.x, voxelSize.y, voxelSize.z);
		corners[7] = voxelPos + make_double3(voxelSize.x, 0, voxelSize.z);

		double sdf[8];
		for (int i = 0; i < 8; ++i)
		{
			sdf[i] = computeSDF(numNodeVerts, d_nodeVertexArray, d_svoNodeArray, d_lambda, corners[i]);
			d_voxelSDF[tid * 8 + i] = sdf[i];
			printf("sdf[%d] = %lf\n", i, sdf[i]);
		}

		int cubeIndex = 0;
		cubeIndex = (uint(sdf[0] < isoVal)) | (uint(sdf[1] < isoVal) << 1) |
			(uint(sdf[2] < isoVal) << 2) | (uint(sdf[3] < isoVal) << 3) |
			(uint(sdf[4] < isoVal) << 4) | (uint(sdf[5] < isoVal) << 5) |
			(uint(sdf[6] < isoVal) << 6) | (uint(sdf[7] < isoVal) << 7);

		int nVerts = tex1Dfetch<int>(nVertsTex, cubeIndex);
		d_nVoxelVertsArray[tid] = nVerts;
		d_isValidVoxelArray[tid] = nVerts > 0;
		d_voxelCubeIndex[tid] = cubeIndex;
	}
}

/**
 * @brief compact voxel array
 *
 * @param nVoxels               voxel的总数量 = res_x * res_y * res_z
 * @param d_isValidVoxel        判断每个 voxel 是否是合理的 voxel
 * @param d_nValidVoxelsScan    exclusive sum of d_isValidVoxel
 * @param d_compactedVoxelArray output
 */
__global__ void MCKernel::compactVoxels(const uint nVoxels,
	const uint* d_isValidVoxel,
	const uint* d_nValidVoxelsScan,
	uint* d_compactedVoxelArray) {
	uint bid = blockIdx.y * gridDim.x + blockIdx.x;
	uint tid = bid * blockDim.x + threadIdx.x;

	if (tid < nVoxels && d_isValidVoxel[tid])
		d_compactedVoxelArray[d_nValidVoxelsScan[tid]] = tid;
}

/**
 * @brief 根据每个体素的 sdf 分布转为 mesh
 *
 * @param maxVerts              MC算法包含的最多的可能点数量
 * @param nValidVoxels          合理的 voxel的总数量 = res_x * res_y * res_z
 * @param d_voxelSize           每个 voxel 的大小
 * @param d_isoVal              isosurface value
 * @param d_origin              MC算法被执行的初始区域原点坐标
 * @param d_res                 分辨率
 * @param d_compactedVoxelArray 去除 invalid 的 voxel 数组
 * @param d_nVoxelVerts         经过 cube index 映射后每个 voxel 内应含点的数量
 * @param d_voxelCubeIndex      每个 voxel 内 sdf 分布所对应的 cube index
 * @param d_voxelSDF            每个 voxel 八个顶点的 sdf 值
 * @param d_nVertsScanned       所有合理 voxel 的点数量前缀和
 * @param d_triPoints           输出，保存实际 mesh 的所有点位置
 */
__global__ void MCKernel::voxelToMeshKernel(const uint nValidVoxels, const uint maxVerts, const double* d_isoVal,
	const double3* d_voxelSize, const double3* d_origin, const uint3* d_res,
	const uint* d_compactedVoxelArray, const cudaTextureObject_t nVertsTex,
	const cudaTextureObject_t triTex, uint* d_voxelCubeIndex,
	double* d_voxelSDF, uint* d_nVertsScanned, double3* d_triPoints) {
	uint bid = blockIdx.y * gridDim.x + blockIdx.x;
	uint tid = bid * blockDim.x + threadIdx.x;

	if (tid < nValidVoxels) {
		uint voxelIdx = d_compactedVoxelArray[tid];

		double isoVal = *d_isoVal;

		uint3 voxelShift = getVoxelShift(voxelIdx, *d_res);
		double3 voxelPos; // voxel 原点坐标
		double3 voxelSize = *d_voxelSize;

		voxelPos.x = voxelShift.x * voxelSize.x;
		voxelPos.y = voxelShift.y * voxelSize.y;
		voxelPos.z = voxelShift.z * voxelSize.z;
		voxelPos += (*d_origin);

		uint cubeIndex = d_voxelCubeIndex[voxelIdx];
		double sdf[8];
		for (int i = 0; i < 8; ++i)
			sdf[i] = d_voxelSDF[voxelIdx * 8 + i];

		double3 corners[8];
		corners[0] = voxelPos;
		corners[1] = voxelPos + make_double3(0, voxelSize.y, 0);
		corners[2] = voxelPos + make_double3(voxelSize.x, voxelSize.y, 0);
		corners[3] = voxelPos + make_double3(voxelSize.x, 0, 0);
		corners[4] = voxelPos + make_double3(0, 0, voxelSize.z);
		corners[5] = voxelPos + make_double3(0, voxelSize.y, voxelSize.z);
		corners[6] = voxelPos + make_double3(voxelSize.x, voxelSize.y, voxelSize.z);
		corners[7] = voxelPos + make_double3(voxelSize.x, 0, voxelSize.z);

		// 预防线程束分化，12条边全都计算一次插值点，反正最后三角形排列方式也是由
		// cube index 决定
		double3 triVerts[12];
		triVerts[0] = vertexLerp(corners[0], corners[1], sdf[0], sdf[1], isoVal);
		triVerts[1] = vertexLerp(corners[1], corners[2], sdf[1], sdf[2], isoVal);
		triVerts[2] = vertexLerp(corners[2], corners[3], sdf[2], sdf[3], isoVal);
		triVerts[3] = vertexLerp(corners[3], corners[0], sdf[3], sdf[0], isoVal);

		triVerts[4] = vertexLerp(corners[4], corners[5], sdf[4], sdf[5], isoVal);
		triVerts[5] = vertexLerp(corners[5], corners[6], sdf[5], sdf[6], isoVal);
		triVerts[6] = vertexLerp(corners[6], corners[7], sdf[6], sdf[7], isoVal);
		triVerts[7] = vertexLerp(corners[7], corners[4], sdf[7], sdf[4], isoVal);

		triVerts[8] = vertexLerp(corners[0], corners[4], sdf[0], sdf[4], isoVal);
		triVerts[9] = vertexLerp(corners[1], corners[5], sdf[1], sdf[5], isoVal);
		triVerts[10] = vertexLerp(corners[2], corners[6], sdf[2], sdf[6], isoVal);
		triVerts[11] = vertexLerp(corners[3], corners[7], sdf[3], sdf[7], isoVal);

		int nVerts = tex1Dfetch<int>(nVertsTex, cubeIndex);

		for (int i = 0; i < nVerts; i += 3) {
			uint triPosIndex = d_nVertsScanned[voxelIdx] + i;

			double3 triangle[3];

			int edgeIndex = tex1Dfetch<int>(triTex, (cubeIndex * 16) + i);
			triangle[0] = triVerts[edgeIndex];

			edgeIndex = tex1Dfetch<int>(triTex, (cubeIndex * 16) + i + 1);
			triangle[1] = triVerts[edgeIndex];

			edgeIndex = tex1Dfetch<int>(triTex, (cubeIndex * 16) + i + 2);
			triangle[2] = triVerts[edgeIndex];

			if (triPosIndex < maxVerts - 3) {
				d_triPoints[triPosIndex] = triangle[0];
				d_triPoints[triPosIndex + 1] = triangle[1];
				d_triPoints[triPosIndex + 2] = triangle[2];
			}
		}
	}
}

inline void MC::d_thrustExclusiveScan(const uint& nElems, uint* input,
	uint* output) {
	thrust::exclusive_scan(thrust::device_ptr<uint>(input),
		thrust::device_ptr<uint>(input + nElems),
		thrust::device_ptr<uint>(output));
}

inline void MC::setTextureObject(const uint& srcSizeInBytes, int* srcDev,
	cudaTextureObject_t* texObj) {
	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

	cudaResourceDesc texRes;
	cudaTextureDesc texDesc;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	memset(&texDesc, 0, sizeof(cudaTextureDesc));

	texRes.resType = cudaResourceTypeLinear;
	texRes.res.linear.devPtr = srcDev;
	texRes.res.linear.sizeInBytes = srcSizeInBytes;
	texRes.res.linear.desc = channelDesc;

	texDesc.normalizedCoords = false;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.readMode = cudaReadModeElementType;

	CUDA_CHECK(cudaCreateTextureObject(texObj, &texRes, &texDesc, nullptr));
}

inline void MC::initCommonResources(const uint& nVoxels,
	const uint3& resolution, const double& isoVal, const double3& gridOrigin,
	const double3& voxelSize, const uint& maxVerts) {
	// host
		{
			h_triPoints = (double3*)malloc(sizeof(double3) * maxVerts);
		}

		// device
		{
			CUDA_CHECK(cudaMalloc((void**)&d_res, sizeof(uint3)));
			CUDA_CHECK(cudaMemcpy(d_res, &resolution, sizeof(uint3), cudaMemcpyHostToDevice));

			CUDA_CHECK(cudaMalloc((void**)&d_isoVal, sizeof(double)));
			CUDA_CHECK(cudaMemcpy(d_isoVal, &isoVal, sizeof(double), cudaMemcpyHostToDevice));

			CUDA_CHECK(cudaMalloc((void**)&d_gridOrigin, sizeof(double3)));
			CUDA_CHECK(cudaMemcpy(d_gridOrigin, &gridOrigin, sizeof(double3), cudaMemcpyHostToDevice));

			CUDA_CHECK(cudaMalloc((void**)&d_voxelSize, sizeof(double3)));
			CUDA_CHECK(cudaMemcpy(d_voxelSize, &voxelSize, sizeof(double3), cudaMemcpyHostToDevice));

			CUDA_CHECK(cudaMalloc((void**)&d_triTable, sizeof(int) * 256 * 16));
			CUDA_CHECK(cudaMemcpy(d_triTable, triTable, sizeof(int) * 256 * 16, cudaMemcpyHostToDevice));

			CUDA_CHECK(cudaMalloc((void**)&d_nVertsTable, sizeof(int) * 256));
			CUDA_CHECK(cudaMemcpy(d_nVertsTable, nVertsTable, sizeof(int) * 256, cudaMemcpyHostToDevice));

			// texture
			setTextureObject(256 * 16 * sizeof(int), d_triTable, &triTex);
			setTextureObject(256 * sizeof(int), d_nVertsTable, &nVertsTex);
		}
}

inline void MC::freeCommonResources() {
	// host
	{ free(h_triPoints); }

	// device
	{
		CUDA_CHECK(cudaFree(d_res));
		CUDA_CHECK(cudaFree(d_isoVal));

		CUDA_CHECK(cudaFree(d_gridOrigin));
		CUDA_CHECK(cudaFree(d_voxelSize));

		CUDA_CHECK(cudaFree(d_triTable));
		CUDA_CHECK(cudaFree(d_nVertsTable));

		// texture object
		CUDA_CHECK(cudaDestroyTextureObject(triTex));
		CUDA_CHECK(cudaDestroyTextureObject(nVertsTex));
	}
}

inline void MC::launch_computSDFKernel(const uint& nVoxels,
	const uint& numNodes, const size_t& _numNodeVerts,
	const VXd& lambda, const std::vector<V3d>& nodeWidthArray,
	const vector<vector<thrust::pair<Eigen::Vector3d, uint32_t>>>& depthNodeVertexArray)
{
	// init resources
	numNodeVerts = _numNodeVerts;
	CUDA_CHECK(cudaMalloc((void**)&d_nodeVertexArray, sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * numNodeVerts));
	size_t offset = 0;
	for (int i = 0; i < depthNodeVertexArray.size(); ++i)
	{
		const size_t i_nodeVerts = depthNodeVertexArray[i].size();
		CUDA_CHECK(cudaMemcpy(d_nodeVertexArray + offset, depthNodeVertexArray[i].data(),
			sizeof(thrust::pair<Eigen::Vector3d, uint32_t>) * i_nodeVerts, cudaMemcpyHostToDevice));
		offset += i_nodeVerts;
	}
	CUDA_CHECK(cudaMalloc((void**)&d_lambda, sizeof(double) * lambda.rows()));
	CUDA_CHECK(cudaMemcpy(d_lambda, lambda.data(), sizeof(double) * lambda.rows(), cudaMemcpyHostToDevice)); // lambda.rows() == numNodeVerts
	CUDA_CHECK(cudaMalloc((void**)&d_nodeWidthArray, sizeof(V3d) * numNodes));
	CUDA_CHECK(cudaMemcpy(d_nodeWidthArray, nodeWidthArray.data(), sizeof(V3d) * numNodes, cudaMemcpyHostToDevice));
	d_voxelSDF.resize(nVoxels * 8);

	// device
	cudaDeviceProp prop;
	int device = getMaxComputeDevice();
	CUDA_CHECK(cudaGetDevice(&device));
	CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

	using namespace std::chrono;
	time_point<system_clock> start, end;

	// streams
	cudaStream_t streams[MAX_NUM_STREAMS];
	for (int i = 0; i < MAX_NUM_STREAMS; ++i) CUDA_CHECK(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

	for (int i = 0; i < MAX_NUM_STREAMS; ++i) {
		uint voxelElems = (nVoxels + MAX_NUM_STREAMS - 1) / MAX_NUM_STREAMS;
		uint voxelOffset = i * voxelElems;
		voxelElems = voxelOffset + voxelElems > nVoxels ? nVoxels - voxelOffset : voxelElems;

		uint* d_voxelOffset = nullptr;
		CUDA_CHECK(cudaMalloc((void**)&d_voxelOffset, sizeof(uint)));
		CUDA_CHECK(cudaMemcpy(d_voxelOffset, &voxelOffset, sizeof(uint), cudaMemcpyHostToDevice));

		const uint numVoxelElemCorners = voxelElems * 8;
		thrust::device_vector<V3d> d_voxelCornerData(numVoxelElemCorners);

		int minGridSize, blockSize, gridSize;
		getOccupancyMaxPotentialBlockSize(nVoxels, minGridSize, blockSize, gridSize, MCKernel::prepareVoxelCornerKernel);
		MCKernel::prepareVoxelCornerKernel << <gridSize, blockSize >> > (voxelElems, d_voxelOffset, d_gridOrigin, d_voxelSize, d_res, d_voxelCornerData.data().get());
		getLastCudaError("Kernel: 'determineVoxelKernel' launch failed!\n");

		thrust::device_vector<double> d_voxelElemSDF(numVoxelElemCorners);
		cuAcc::cpBSplineVal(prop, numVoxelElemCorners, numNodeVerts, d_nodeVertexArray, d_nodeWidthArray, d_lambda, d_voxelCornerData, d_voxelElemSDF, streams[i]);
		CUDA_CHECK(cudaMemcpyAsync((d_voxelSDF.data() + voxelOffset * 8).get(), d_voxelElemSDF.data().get(), sizeof(double) * numVoxelElemCorners, cudaMemcpyDeviceToDevice, streams[i]));

		cleanupThrust(d_voxelElemSDF);
		cleanupThrust(d_voxelCornerData);
		CUDA_CHECK(cudaFree(d_voxelOffset));
	}

	for (int i = 0; i < MAX_NUM_STREAMS; i++)
		cudaStreamSynchronize(streams[i]);
	for (int i = 0; i < MAX_NUM_STREAMS; ++i)
		CUDA_CHECK(cudaStreamDestroy(streams[i]));

	// free resources
	CUDA_CHECK(cudaFree(d_lambda));
	CUDA_CHECK(cudaFree(d_nodeWidthArray));
	CUDA_CHECK(cudaFree(d_nodeVertexArray));

	h_voxelSDF = d_voxelSDF;
}

inline void MC::launch_determineVoxelKernel(const uint& nVoxels,
	 const double& isoVal,
	 const uint& maxVerts)
{
	// init resources
	CUDA_CHECK(cudaMalloc((void**)&d_voxelCubeIndex, sizeof(uint) * nVoxels));
	CUDA_CHECK(cudaMalloc((void**)&d_nVoxelVertsArray, sizeof(uint) * nVoxels));
	CUDA_CHECK(cudaMalloc((void**)&d_nVoxelVertsScan, sizeof(uint) * nVoxels));
	CUDA_CHECK(cudaMalloc((void**)&d_isValidVoxelArray, sizeof(uint) * nVoxels));
	CUDA_CHECK(cudaMalloc((void**)&d_nValidVoxelsScan, sizeof(uint) * nVoxels));

	dim3 nThreads(V_NTHREADS, 1, 1);
	dim3 nBlocks((nVoxels + nThreads.x - 1) / nThreads.x, 1, 1);
	while (nBlocks.x > 65535) {
		nBlocks.x /= 2;
		nBlocks.y *= 2;
	}

	MCKernel::determineVoxelKernel << <nBlocks, nThreads >> > (nVoxels, d_isoVal, nVertsTex, d_nVoxelVertsArray,
		d_voxelCubeIndex, d_voxelSDF.data().get(), d_isValidVoxelArray);
	/*MCKernel::determineVoxelKernel_2 << <nBlocks, nThreads >> > (nVoxels, numNodeVerts, d_isoVal, d_gridOrigin,
		d_voxelSize, d_res, d_nodeVertexArray, d_svoNodeArray.data().get(), d_lambda, nVertsTex, d_nVoxelVertsArray,
		d_voxelCubeIndex, d_voxelSDF.data().get(), d_isValidVoxelArray);*/
	getLastCudaError("Kernel: 'determineVoxelKernel' launch failed!\n");

	d_thrustExclusiveScan(nVoxels, d_nVoxelVertsArray, d_nVoxelVertsScan);
	d_thrustExclusiveScan(nVoxels, d_isValidVoxelArray, d_nValidVoxelsScan);

	uint lastElement, lastScanElement;
	CUDA_CHECK(cudaMemcpy(&lastElement, d_isValidVoxelArray + nVoxels - 1,
		sizeof(uint), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&lastScanElement, d_nValidVoxelsScan + nVoxels - 1,
		sizeof(uint), cudaMemcpyDeviceToHost));
	nValidVoxels = lastElement + lastScanElement;
	if (nValidVoxels == 0) return;

	CUDA_CHECK(cudaMemcpy(&lastElement, d_nVoxelVertsArray + nVoxels - 1,
		sizeof(uint), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&lastScanElement, d_nVoxelVertsScan + nVoxels - 1,
		sizeof(uint), cudaMemcpyDeviceToHost));
	allTriVertices = lastElement + lastScanElement;

	// free resources
	CUDA_CHECK(cudaFree(d_nVoxelVertsArray));
}

inline void MC::launch_compactVoxelsKernel(const uint& nVoxels)
{
	// init resources
	CUDA_CHECK(cudaMalloc((void**)&d_compactedVoxelArray, sizeof(uint) * nVoxels));

	dim3 nThreads(V_NTHREADS, 1, 1);
	dim3 nBlocks((nVoxels + nThreads.x - 1) / nThreads.x, 1, 1);
	while (nBlocks.x > 65535) {
		nBlocks.x /= 2;
		nBlocks.y *= 2;
	}

	MCKernel::compactVoxels << <nBlocks, nThreads >> > (
		nVoxels, d_isValidVoxelArray, d_nValidVoxelsScan, d_compactedVoxelArray);
	getLastCudaError("Kernel: 'compactVoxelsKernel' launch failed!\n");

	// free resources
	CUDA_CHECK(cudaFree(d_isValidVoxelArray));
	CUDA_CHECK(cudaFree(d_nValidVoxelsScan));
}

inline void MC::launch_voxelToMeshKernel(const uint& maxVerts,
	const uint& nVoxels)
{
	// init resources
	CUDA_CHECK(cudaMalloc((void**)&d_triPoints, sizeof(double3) * maxVerts));

	dim3 nThreads(V_NTHREADS, 1, 1);
	dim3 nBlocks((nValidVoxels + nThreads.x - 1) / nThreads.x, 1, 1);
	while (nBlocks.x > 65535) {
		nBlocks.x /= 2;
		nBlocks.y *= 2;
	}

	MCKernel::voxelToMeshKernel << <nBlocks, nThreads >> > (
		nValidVoxels, maxVerts, d_isoVal, d_voxelSize, d_gridOrigin, d_res,
		d_compactedVoxelArray, nVertsTex, triTex, d_voxelCubeIndex, d_voxelSDF.data().get(),
		d_nVoxelVertsScan, d_triPoints);
	getLastCudaError("Kernel: 'determineVoxelKernel' launch failed!\n");

	CUDA_CHECK(cudaMemcpy(h_triPoints, d_triPoints, sizeof(double3) * maxVerts,
		cudaMemcpyDeviceToHost));

	// free resources
	cleanupThrust(d_voxelSDF);
	CUDA_CHECK(cudaFree(d_compactedVoxelArray));
	CUDA_CHECK(cudaFree(d_voxelCubeIndex));
	CUDA_CHECK(cudaFree(d_nVoxelVertsScan));
	CUDA_CHECK(cudaFree(d_triPoints));
}

inline void MC::writeToOBJFile(const std::string& filename) {
	checkDir(filename);
	std::ofstream out(filename);
	if (!out) {
		fprintf(stderr, "IO Error: File %s could not be opened!\n",
			filename.c_str());
		return;
	}

	printf("-- The number of mesh's vertices = %d\n", allTriVertices);
	printf("-- The number of mesh's faces = %d\n", allTriVertices / 3);
	for (int i = 0; i < allTriVertices; i += 3) {
		const int faceIdx = i;

		out << "v " << h_triPoints[i].x << ' ' << h_triPoints[i].y << ' '
			<< h_triPoints[i].z << '\n';
		out << "v " << h_triPoints[i + 1].x << ' ' << h_triPoints[i + 1].y << ' '
			<< h_triPoints[i + 1].z << '\n';
		out << "v " << h_triPoints[i + 2].x << ' ' << h_triPoints[i + 2].y << ' '
			<< h_triPoints[i + 2].z << '\n';

		out << "f " << faceIdx + 1 << ' ' << faceIdx + 2 << ' ' << faceIdx + 3
			<< '\n';
	}

	out.close();
}

/**
 * @brief 主方法
 *
 * @param gridOrigin origin coordinate of grid
 * @param gridWidth  width of grid
 * @param resolution voxel 的分辨率
 * @param isoVal     isosurface value
 * @param filename   output obj file
 */
void MC::marching_cubes(const vector<vector<thrust::pair<Eigen::Vector3d, uint32_t>>>& depthNodeVertexArray,
	const vector<SVONode>& svoNodeArray, const vector<size_t>& esumDepthNodeVerts,
	const size_t& numNodes, const std::vector<V3d>& nodeWidthArray,
	const size_t& numNodeVerts, const VXd& lambda, const double3& gridOrigin, const double3& gridWidth,
	const uint3& resolution, const double& isoVal, const std::string& filename) 
{
	if (numNodeVerts == 0) { printf("[MC] Warning: There is no valid Sparse Voxel Octree's node vertex, MarchingCubes is exited...\n"); return; }
	uint nVoxels = resolution.x * resolution.y * resolution.z;

	uint maxVerts = nVoxels * 18;

	double3 voxelSize = make_double3(gridWidth.x / resolution.x, gridWidth.y / resolution.y, gridWidth.z / resolution.z);

	using namespace std::chrono;
	time_point<system_clock> start, end;

	start = system_clock::now();

	initCommonResources(nVoxels, resolution, isoVal, gridOrigin, voxelSize, maxVerts);

	if (h_voxelSDF.empty()) launch_computSDFKernel(nVoxels, numNodes, numNodeVerts, lambda, nodeWidthArray, depthNodeVertexArray);
	else d_voxelSDF = h_voxelSDF;

	launch_determineVoxelKernel(nVoxels, isoVal, maxVerts);
	if (allTriVertices == 0) {
		printf("-- MC: There is no valid vertices...\n");
		return;
	}

	launch_compactVoxelsKernel(nVoxels);

	launch_voxelToMeshKernel(maxVerts, nVoxels);

	end = system_clock::now();
	duration<double> elapsed_seconds = end - start;
	std::time_t end_time = system_clock::to_time_t(end);
	std::cout << "-- MarchingCubes finished computation at " << std::ctime(&end_time)
		<< "-- Elapsed time: " << elapsed_seconds.count() << " s\n";

	std::cout << "-- Write to obj..." << std::endl;
	writeToOBJFile(filename);

	freeCommonResources();
}
