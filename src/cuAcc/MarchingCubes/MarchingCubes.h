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
#include "..\..\Octree.h"
#include "..\..\utils\cuda\CUDACheck.cuh"
#include "..\..\utils\cuda\CUDAMath.hpp"

namespace MCKernel {
	template <typename T> inline __host__ __device__ T lerp(T v0, T v1, T t) {
		return fma(t, v1, fma(-t, v0, v0));
	}

	__device__ double3 vertexLerp(const double3& p_0, const double3& p_1,
		const double& sdf_0, const double& sdf_1,
		const double& isoVal);

	__device__ double computeSDF(const uint numNodes, double3 pos, double* d_lambda, V3d* d_nodeCorners, V3d* d_nodeWidth);

	__device__ uint3 getVoxelShift(const uint& index, const uint3& d_res);

	__global__ void prepareMatrixKernel(const uint nVoxelElems, const uint nAllNodes, 
		const uint* d_voxelOffset, const uint3* d_res, double* d_lambda, 
		double3* d_origin, double3* d_voxelSize,
		V3d* d_nodeCorners, V3d* d_nodeWidth, double* d_voxelMatrix);

	__global__ void
		determineVoxelKernel(const uint nVoxels, const double* d_isoVal,
			const cudaTextureObject_t nVertsTex,
			uint* d_nVoxelVerts, uint* d_voxelCubeIndex,
			double* d_voxelSDF, uint* d_isValidVoxelArray);

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
	namespace {
		uint nAllNodes = 0;

		uint allTriVertices = 0, nValidVoxels = 0;

		double3* h_triPoints = nullptr; // output
	} // namespace

	// device
	namespace {
		V3d* d_nodeCorners = nullptr;
		V3d* d_nodeWidth = nullptr;

		double* d_lambda = nullptr;

		uint3* d_res = nullptr;
		double* d_isoVal = nullptr;

		uint* d_nVoxelVertsArray = nullptr;
		uint* d_nVoxelVertsScan = nullptr;

		uint* d_isValidVoxelArray = nullptr;
		uint* d_nValidVoxelsScan = nullptr;

		double3* d_gridOrigin = nullptr;
		double3* d_voxelSize = nullptr;

		double* d_voxelSDF = nullptr;
		uint* d_voxelCubeIndex = nullptr;

		uint* d_compactedVoxelArray = nullptr;

		int* d_triTable = nullptr;
		int* d_nVertsTable = nullptr;

		// textures containing look-up tables
		cudaTextureObject_t triTex;
		cudaTextureObject_t nVertsTex;

		double3* d_triPoints = nullptr; // output
	} // namespace
} // namespace MC

namespace MC {
	void d_thrustExclusiveScan(const uint& nElems, uint* input, uint* output);

	void setTextureObject(const uint& srcSizeInBytes, int* srcDev,
		cudaTextureObject_t* texObj);

	void initResources(const vector<OctreeNode*> allNodes, const VXd& lambda,
		const uint3& resolution, const uint& nVoxels,
		const double& isoVal, const double3& gridOrigin,
		const double3& voxelSize, const uint& maxVerts);

	void freeResources();

	void launch_prepareMatrixKernel(const uint& nVoxelElems, const uint& nAllNodes, const uint& voxelOffset, const cudaStream_t& stream, double* d_voxelMatrix);

	void launch_computSDFKernel(const uint& nVoxels, const uint& nAllNodes);

	void launch_determineVoxelKernel(const uint& nVoxels, const double& isoVal, const uint& maxVerts);

	void launch_compactVoxelsKernel(const uint& nVoxels);

	void launch_voxelToMeshKernel(const uint& maxVerts, const uint& nVoxels);

	void writeToOBJFile(const std::string& filename);

	void marching_cubes(const vector<OctreeNode*> allNodes, const VXd& lambda,
		const double3& gridOrigin, const double3& gridWidth,
		const uint3& resolution, const double& isoVal, const std::string& filename);
} // namespace MC
