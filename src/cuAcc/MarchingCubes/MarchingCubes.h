/*
 * @Author: Poet 602289550@qq.com
 * @Date: 2023-03-18 11:31:45
 * @LastEditors: Poet 602289550@qq.com
 * @LastEditTime: 2023-03-28 13:55:57
 * @FilePath: \GPUMarchingCubes\MarchingCubes.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置
 * 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
#include "..\..\SVO.h"
#include "..\..\utils\cuda\CUDACheck.cuh"
#include "..\..\utils\cuda\CUDAMath.hpp"
#include <thrust\host_vector.h>

namespace MCKernel {
	template <typename T>
	__host__ __device__ inline T lerp(T v0, T v1, T t) {
		return fma(t, v1, fma(-t, v0, v0));
	}

	__device__ double3 vertexLerp(const double3 p_0, const double3 p_1,
		const double sdf_0, const double sdf_1,
		const double isoVal);

	__device__ uint3 getVoxelShift(const uint index, const uint3 d_res);

	// 与SDFCompute.cu二选一
	__device__ double computeSDF(const uint numNodeVerts, const thrust::pair<Eigen::Vector3d, uint32_t>* d_nodeVertexArray,
		const SVONode* d_svoNodeArray, const double* d_lambda, double3 pos);

	/*__global__ void prepareMatrixKernel(const uint nVoxelElemCorners, const uint nAllNodeCorners,
		const uint* d_voxelOffset, const uint3* d_res, double* d_lambda,
		double3* d_origin, double3* d_voxelSize,
		V3d* d_nodeCorners, V3d* d_nodeWidth, double* d_voxelMatrix);*/

	__global__ void prepareVoxelCornerKernel(const uint nVoxels, const uint* voxel_offset, const double3* d_origin,
		const double3* d_voxelSize, const uint3* d_res, V3d* d_voxelCornerData);

	__global__ void determineVoxelKernel(const uint nVoxels, const double* d_isoVal,
			const cudaTextureObject_t nVertsTex,
			uint* d_nVoxelVerts, uint* d_voxelCubeIndex,
			double* d_voxelSDF, uint* d_isValidVoxelArray);

	__global__ void determineVoxelKernel_2(const uint nVoxels, const uint nAllNodes, const double* d_isoVal,
		const double3* d_origin, const double3* d_voxelSize, const uint3* d_res, const thrust::pair<Eigen::Vector3d, uint32_t>* nodeVertexArray,
		const SVONode* svoNodeArray, const double* d_lambda, const cudaTextureObject_t nVertsTex, uint* d_nVoxelVertsArray,
		uint* d_voxelCubeIndex, double* d_voxelSDF, uint* d_isValidVoxelArray);

	__global__ void compactVoxels(const uint nVoxels, const uint* d_isValidVoxel,
		const uint* d_nValidVoxelsScan,
		uint* d_compactedVoxelArray);

	__global__ void voxelToMeshKernel(
		const uint nValidVoxels, const uint maxVerts, const double* d_isoVal,
		const double3* d_voxelSize, const double3* d_origin, const uint3* d_res,
		const uint* d_compactedVoxelArray, const cudaTextureObject_t nVertsTex,
		const cudaTextureObject_t triTex, uint* d_voxelCubeIndex,
		double* d_voxelSDF, uint* d_nVertsScanned, double3* d_triPoints);
} // namespace MCKernel

namespace MC {
	// host
	//namespace {
	extern uint numNodeVerts;

	extern uint allTriVertices, nValidVoxels;

	extern double3* h_triPoints; // output
	//} // namespace

	// device
	//namespace {
	extern double* d_lambda;

	extern uint3* d_res;
	extern double* d_isoVal;

	extern uint* d_nVoxelVertsArray;
	extern uint* d_nVoxelVertsScan;

	extern uint* d_isValidVoxelArray;
	extern uint* d_nValidVoxelsScan;

	extern double3* d_gridOrigin;
	extern double3* d_voxelSize;

	extern thrust::device_vector<double> d_voxelSDF;
	extern thrust::host_vector<double> h_voxelSDF;

	extern uint* d_voxelCubeIndex;

	extern uint* d_compactedVoxelArray;

	extern int* d_triTable;
	extern int* d_nVertsTable;

	// textures containing look-up tables
	extern cudaTextureObject_t triTex;
	extern cudaTextureObject_t nVertsTex;

	extern double3* d_triPoints; // output
	//} // namespace
} // namespace MC

namespace MC {
	void d_thrustExclusiveScan(const uint& nElems, uint* input, uint* output);

	void setTextureObject(const uint& srcSizeInBytes, int* srcDev,
		cudaTextureObject_t* texObj);

	void initCommonResources(const uint& nVoxels,
		const uint3& resolution, const double& isoVal, const double3& gridOrigin,
		const double3& voxelSize, const uint& maxVerts);

	void freeCommonResources();

	/*void launch_prepareMatrixKernel(const uint& nVoxelElems, const uint& nAllNodes, const uint& voxelOffset,
		const cudaStream_t& stream, double*& d_voxelMatrix);*/

	void launch_computSDFKernel(const uint& nVoxels,
		const uint& numNodes, const size_t& _numNodeVerts,
		const VXd& lambda, const std::vector<V3d>& nodeWidthArray,
		const vector<vector<thrust::pair<Eigen::Vector3d, uint32_t>>>& depthNodeVertexArray);

	void launch_determineVoxelKernel(const uint& nVoxels, const double& isoVal, const uint& maxVerts);

	void launch_compactVoxelsKernel(const uint& nVoxels);

	void launch_voxelToMeshKernel(const uint& maxVerts, const uint& nVoxels);

	void writeToOBJFile(const std::string& filename);

	void marching_cubes(const vector<vector<thrust::pair<Eigen::Vector3d, uint32_t>>>& depthNodeVertexArray,
		const vector<SVONode>& svoNodeArray, const vector<size_t>& esumDepthNodeVerts,
		const size_t& numNodes, const std::vector<V3d>& nodeWidthArray,
		const size_t& numNodeVerts, const VXd& lambda, const double3& gridOrigin, const double3& gridWidth,
		const uint3& resolution, const double& isoVal, const std::string& filename);
} // namespace MC
