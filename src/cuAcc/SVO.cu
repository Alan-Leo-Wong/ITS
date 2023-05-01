#include "..\SVO.h"
#include "..\MortonLUT.h"
#include "..\utils\IO.hpp"
#include "..\utils\Geometry.hpp"
#include "..\utils\cuda\CUDAMath.hpp"
#include "..\utils\cuda\CUDAUtil.cuh"
#include "..\utils\cuda\CUDACheck.cuh"
#include <thrust\scan.h>
#include <thrust\sort.h>
#include <thrust\unique.h>
#include <thrust\extrema.h>
#include <thrust\device_vector.h>
#include <cuda_runtime_api.h>
#include <iomanip>

namespace {
	// Estimate best block and grid size using CUDA Occupancy Calculator
	int blockSize;   // The launch configurator returned block size 
	int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize;    // The actual grid size needed, based on input size 
}

template <typename T>
struct scanMortonFlag : public thrust::unary_function<T, T> {
	__host__ __device__ T operator()(const T& x) {
		// printf("%lu %d\n", b, (b >> 31) & 1);
		return (x >> 31) & 1;
	}
};

__global__ void surfaceVoxelize(const int nTris,
	const Eigen::Vector3i* d_surfaceVoxelGridSize,
	const Eigen::Vector3d* d_gridOrigin,
	const Eigen::Vector3d* d_unitVoxelSize,
	double* d_triangle_data,
	uint32_t* d_voxelArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
	size_t stride = blockDim.x * gridDim.x;

	const Eigen::Vector3i surfaceVoxelGridSize = *d_surfaceVoxelGridSize;
	const Eigen::Vector3d unitVoxelSize = *d_unitVoxelSize;
	const Eigen::Vector3d gridOrigin = *d_gridOrigin;
	Eigen::Vector3d delta_p{ unitVoxelSize.x(), unitVoxelSize.y(), unitVoxelSize.z() };
	Eigen::Vector3i grid_max{ surfaceVoxelGridSize.x() - 1, surfaceVoxelGridSize.y() - 1, surfaceVoxelGridSize.z() - 1 }; // grid max (grid runs from 0 to gridsize-1)
	while (tid < nTris) { // every thread works on specific triangles in its stride
		size_t t = tid * 9; // triangle contains 9 vertices

		// COMPUTE COMMON TRIANGLE PROPERTIES
		// Move vertices to origin using modelBBox
		Eigen::Vector3d v0 = Eigen::Vector3d(d_triangle_data[t], d_triangle_data[t + 1], d_triangle_data[t + 2]) - gridOrigin;
		Eigen::Vector3d v1 = Eigen::Vector3d(d_triangle_data[t + 3], d_triangle_data[t + 4], d_triangle_data[t + 5]) - gridOrigin;
		Eigen::Vector3d v2 = Eigen::Vector3d(d_triangle_data[t + 6], d_triangle_data[t + 7], d_triangle_data[t + 8]) - gridOrigin;
		// Edge vectors
		Eigen::Vector3d e0 = v1 - v0;
		Eigen::Vector3d e1 = v2 - v1;
		Eigen::Vector3d e2 = v0 - v2;
		// Normal vector pointing up from the triangle
		Eigen::Vector3d n = e0.cross(e1).normalized();

		// COMPUTE TRIANGLE BBOX IN GRID
		// Triangle bounding box in world coordinates is min(v0,v1,v2) and max(v0,v1,v2)
		AABox<Eigen::Vector3d> t_bbox_world(fminf(v0, fminf(v1, v2)), fmaxf(v0, fmaxf(v1, v2)));
		// Triangle bounding box in voxel grid coordinates is the world bounding box divided by the grid unit vector
		AABox<Eigen::Vector3i> t_bbox_grid;
		t_bbox_grid.boxOrigin = clamp(
			Eigen::Vector3i((t_bbox_world.boxOrigin.x() / unitVoxelSize.x()), (t_bbox_world.boxOrigin.y() / unitVoxelSize.y()), (t_bbox_world.boxOrigin.z() / unitVoxelSize.z())),
			Eigen::Vector3i(0, 0, 0), grid_max
		);
		t_bbox_grid.boxEnd = clamp(
			Eigen::Vector3i((t_bbox_world.boxEnd.x() / unitVoxelSize.x()), (t_bbox_world.boxEnd.y() / unitVoxelSize.y()), (t_bbox_world.boxEnd.z() / unitVoxelSize.z())),
			Eigen::Vector3i(0, 0, 0), grid_max
		);

		// PREPARE PLANE TEST PROPERTIES
		Eigen::Vector3d c(0.0, 0.0, 0.0);
		if (n.x() > 0.0) { c.x() = unitVoxelSize.x(); }
		if (n.y() > 0.0) { c.y() = unitVoxelSize.y(); }
		if (n.z() > 0.0) { c.z() = unitVoxelSize.z(); }
		double d1 = n.dot((c - v0));
		double d2 = n.dot(((delta_p - c) - v0));

		// PREPARE PROJECTION TEST PROPERTIES
		// XY plane
		Eigen::Vector2d n_xy_e0(-1.0 * e0.y(), e0.x());
		Eigen::Vector2d n_xy_e1(-1.0 * e1.y(), e1.x());
		Eigen::Vector2d n_xy_e2(-1.0 * e2.y(), e2.x());
		if (n.z() < 0.0)
		{
			n_xy_e0 = -n_xy_e0;
			n_xy_e1 = -n_xy_e1;
			n_xy_e2 = -n_xy_e2;
		}
		double d_xy_e0 = (-1.0 * n_xy_e0.dot(Eigen::Vector2d(v0.x(), v0.y()))) + fmaxf(0.0, unitVoxelSize.x() * n_xy_e0[0]) + fmaxf(0.0, unitVoxelSize.y() * n_xy_e0[1]);
		double d_xy_e1 = (-1.0 * n_xy_e1.dot(Eigen::Vector2d(v1.x(), v1.y()))) + fmaxf(0.0, unitVoxelSize.x() * n_xy_e1[0]) + fmaxf(0.0, unitVoxelSize.y() * n_xy_e1[1]);
		double d_xy_e2 = (-1.0 * n_xy_e2.dot(Eigen::Vector2d(v2.x(), v2.y()))) + fmaxf(0.0, unitVoxelSize.x() * n_xy_e2[0]) + fmaxf(0.0, unitVoxelSize.y() * n_xy_e2[1]);
		// YZ plane
		Eigen::Vector2d n_yz_e0(-1.0 * e0.z(), e0.y());
		Eigen::Vector2d n_yz_e1(-1.0 * e1.z(), e1.y());
		Eigen::Vector2d n_yz_e2(-1.0 * e2.z(), e2.y());
		if (n.x() < 0.0) {
			n_yz_e0 = -n_yz_e0;
			n_yz_e1 = -n_yz_e1;
			n_yz_e2 = -n_yz_e2;
		}
		double d_yz_e0 = (-1.0 * n_yz_e0.dot(Eigen::Vector2d(v0.y(), v0.z()))) + fmaxf(0.0, unitVoxelSize.y() * n_yz_e0[0]) + fmaxf(0.0, unitVoxelSize.z() * n_yz_e0[1]);
		double d_yz_e1 = (-1.0 * n_yz_e1.dot(Eigen::Vector2d(v1.y(), v1.z()))) + fmaxf(0.0, unitVoxelSize.y() * n_yz_e1[0]) + fmaxf(0.0, unitVoxelSize.z() * n_yz_e1[1]);
		double d_yz_e2 = (-1.0 * n_yz_e2.dot(Eigen::Vector2d(v2.y(), v2.z()))) + fmaxf(0.0, unitVoxelSize.y() * n_yz_e2[0]) + fmaxf(0.0, unitVoxelSize.z() * n_yz_e2[1]);
		// ZX plane																							 													  
		Eigen::Vector2d n_zx_e0(-1.0 * e0.x(), e0.z());
		Eigen::Vector2d n_zx_e1(-1.0 * e1.x(), e1.z());
		Eigen::Vector2d n_zx_e2(-1.0 * e2.x(), e2.z());
		if (n.y() < 0.0) {
			n_zx_e0 = -n_zx_e0;
			n_zx_e1 = -n_zx_e1;
			n_zx_e2 = -n_zx_e2;
		}
		double d_xz_e0 = (-1.0 * n_zx_e0.dot(Eigen::Vector2d(v0.z(), v0.x()))) + fmaxf(0.0, unitVoxelSize.z() * n_zx_e0[0]) + fmaxf(0.0, unitVoxelSize.x() * n_zx_e0[1]);
		double d_xz_e1 = (-1.0 * n_zx_e1.dot(Eigen::Vector2d(v1.z(), v1.x()))) + fmaxf(0.0, unitVoxelSize.z() * n_zx_e1[0]) + fmaxf(0.0, unitVoxelSize.x() * n_zx_e1[1]);
		double d_xz_e2 = (-1.0 * n_zx_e2.dot(Eigen::Vector2d(v2.z(), v2.x()))) + fmaxf(0.0, unitVoxelSize.z() * n_zx_e2[0]) + fmaxf(0.0, unitVoxelSize.x() * n_zx_e2[1]);

		// test possible grid boxes for overlap
		for (uint16_t z = t_bbox_grid.boxOrigin.z(); z <= t_bbox_grid.boxEnd.z(); z++) {
			for (uint16_t y = t_bbox_grid.boxOrigin.y(); y <= t_bbox_grid.boxEnd.y(); y++) {
				for (uint16_t x = t_bbox_grid.boxOrigin.x(); x <= t_bbox_grid.boxEnd.x(); x++) {
					// if (checkBit(voxel_table, location)){ continue; }
					// TRIANGLE PLANE THROUGH BOX TEST
					Eigen::Vector3d p(x * unitVoxelSize.x(), y * unitVoxelSize.y(), z * unitVoxelSize.z());
					double nDOTp = n.dot(p);
					if ((nDOTp + d1) * (nDOTp + d2) > 0.0) { continue; }

					// PROJECTION TESTS
					// XY
					Eigen::Vector2d p_xy(p.x(), p.y());
					if ((n_xy_e0.dot(p_xy) + d_xy_e0) < 0.0) { continue; }
					if ((n_xy_e1.dot(p_xy) + d_xy_e1) < 0.0) { continue; }
					if ((n_xy_e2.dot(p_xy) + d_xy_e2) < 0.0) { continue; }

					// YZ
					Eigen::Vector2d p_yz(p.y(), p.z());
					if ((n_yz_e0.dot(p_yz) + d_yz_e0) < 0.0) { continue; }
					if ((n_yz_e1.dot(p_yz) + d_yz_e1) < 0.0) { continue; }
					if ((n_yz_e2.dot(p_yz) + d_yz_e2) < 0.0) { continue; }

					// XZ	
					Eigen::Vector2d p_zx(p.z(), p.x());
					if ((n_zx_e0.dot(p_zx) + d_xz_e0) < 0.0) { continue; }
					if ((n_zx_e1.dot(p_zx) + d_xz_e1) < 0.0) { continue; }
					if ((n_zx_e2.dot(p_zx) + d_xz_e2) < 0.0) { continue; }

					uint32_t mortonCode = morton::mortonEncode_LUT(x, y, z);
					atomicExch(d_voxelArray + mortonCode, mortonCode | E_MORTON_32_FLAG); // 最高位设置为1，代表这是个表面的voxel
				}
			}
		}
		tid += stride;
	}
}

void SparseVoxelOctree::meshVoxelize(const size_t& nModelTris,
	const vector<Triangle<V3d>>& modelTris,
	const Eigen::Vector3i* d_surfaceVoxelGridSize,
	const Eigen::Vector3d* d_unitVoxelSize,
	const Eigen::Vector3d* d_gridOrigin,
	thrust::device_vector<uint32_t>& d_CNodeMortonArray)
{
	thrust::device_vector<Eigen::Vector3d> d_triangleThrustVec;
	for (int i = 0; i < nModelTris; ++i)
	{
		d_triangleThrustVec.push_back(modelTris[i].p1);
		d_triangleThrustVec.push_back(modelTris[i].p2);
		d_triangleThrustVec.push_back(modelTris[i].p3);
	}
	double* d_triangleData = (double*)thrust::raw_pointer_cast(&(d_triangleThrustVec[0]));
	getOccupancyMaxPotentialBlockSize(nModelTris, minGridSize, blockSize, gridSize, surfaceVoxelize, 0, 0);
	surfaceVoxelize << <gridSize, blockSize >> > (nModelTris, d_surfaceVoxelGridSize,
		d_gridOrigin, d_unitVoxelSize, d_triangleData, d_CNodeMortonArray.data().get());
	getLastCudaError("Kernel 'surfaceVoxelize' launch failed!\n");
	//cudaDeviceSynchronize();
}


_CUDA_GENERAL_CALL_ uint32_t getParentMorton(const uint32_t morton)
{
	return (((morton >> 3) & 0xfffffff));
}

_CUDA_GENERAL_CALL_ bool isSameParent(const uint32_t morton_1, const uint32_t morton_2)
{
	return getParentMorton(morton_1) == getParentMorton(morton_2);
}

__global__ void compactArray(const int n,
	const bool* d_isValidArray,
	const uint32_t* d_dataArray,
	const size_t* d_esumDataArray,
	uint32_t* d_pactDataArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < n && d_isValidArray[tid])
		d_pactDataArray[d_esumDataArray[tid]] = d_dataArray[tid];
}

// 计算表面voxel共对应多少个八叉树节点同时设置父节点的莫顿码数组
__global__ void cpNumNodes(const size_t n,
	const uint32_t* d_pactDataArray,
	short int* d_nNodesArray,
	uint32_t* d_parentMortonArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= 1 && tid < n)
	{
		if (isSameParent(d_pactDataArray[tid], d_pactDataArray[tid - 1])) d_nNodesArray[tid] = 0;
		else
		{
			const uint32_t parentMorton = getParentMorton(d_pactDataArray[tid]);
			d_parentMortonArray[parentMorton] = parentMorton | E_MORTON_32_FLAG;
			d_nNodesArray[tid] = 8;
		}
	}
}

__global__ void createNode_1(const size_t pactSize,
	const size_t* d_sumNodesArray,
	const uint32_t* d_pactDataArray,
	const Eigen::Vector3d* d_gridOrigin,
	const double* d_width,
	uint32_t* d_begMortonArray,
	SVONode* d_nodeArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	uint16_t x, y, z;
	if (tid < pactSize)
	{
		const Eigen::Vector3d gridOrigin = *d_gridOrigin;
		const double width = *d_width;

		const int sumNodes = d_sumNodesArray[tid];
		const uint32_t pactData = d_pactDataArray[tid];

		const uint32_t key = pactData & LOWER_3BIT_MASK;
		const uint32_t morton = pactData & D_MORTON_32_FLAG; // 去除符号位的实际莫顿码
		// 得到mortonCode对应的实际存储节点的位置
		const size_t address = sumNodes + key;

		SVONode& tNode = d_nodeArray[address];
		tNode.mortonCode = morton;
		morton::morton3D_32_decode(morton, x, y, z);
		tNode.origin = gridOrigin + width * Eigen::Vector3d((double)x, (double)y, (double)z);
		tNode.width = width;

		d_begMortonArray[tid] = (morton / 8) * 8;
	}
}

__global__ void createNode_2(const size_t pactSize,
	const size_t d_preChildDepthTreeNodes, // 子节点层的前面所有层的节点数量(exclusive scan)，用于确定在总节点数组中的位置
	const size_t d_preDepthTreeNodes, // 当前层的前面所有层的节点数量(exclusive scan)，用于确定在总节点数组中的位置
	const size_t* d_sumNodesArray, // 这一层的节点数量inclusive scan数组
	const uint32_t* d_pactDataArray,
	const Eigen::Vector3d* d_gridOrigin,
	const double* d_width,
	uint32_t* d_begMortonArray,
	SVONode* d_nodeArray,
	SVONode* d_childArray)
{
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	uint16_t x, y, z;
	if (tid < pactSize)
	{
		const Eigen::Vector3d gridOrigin = *d_gridOrigin;
		const double width = *d_width;

		const int sumNodes = d_sumNodesArray[tid];
		const uint32_t pactData = d_pactDataArray[tid];

		const uint32_t key = pactData & LOWER_3BIT_MASK;
		const uint32_t morton = pactData & D_MORTON_32_FLAG;
		const size_t address = sumNodes + key;

		SVONode& tNode = d_nodeArray[address];
		tNode.mortonCode = morton;
		morton::morton3D_32_decode(morton, x, y, z);
		tNode.origin = gridOrigin + width * Eigen::Vector3d((double)x, (double)y, (double)z);
		tNode.width = width;
		tNode.isLeaf = false;

		d_begMortonArray[tid] = (morton / 8) * 8;

#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			tNode.childs[i] = d_preChildDepthTreeNodes + tid * 8 + i;
			d_childArray[tid * 8 + i].parent = d_preDepthTreeNodes + tid;
		}
	}
}

__global__ void createRemainNode(const size_t nNodes,
	const Eigen::Vector3d* d_gridOrigin,
	const double* d_width,
	const uint32_t* d_begMortonArray,
	SVONode* d_nodeArray)
{
	extern __shared__ uint32_t sh_begMortonArray[];
	size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	uint16_t x, y, z;
	if (tid < nNodes)
	{
		sh_begMortonArray[threadIdx.x / 8] = d_begMortonArray[tid / 8];

		__syncthreads();

		if (d_nodeArray[tid].mortonCode == 0)
		{
			const Eigen::Vector3d gridOrigin = *d_gridOrigin;
			const double width = *d_width;

			const uint32_t key = tid & LOWER_3BIT_MASK;
			const uint32_t morton = sh_begMortonArray[threadIdx.x / 8] + key;

			SVONode& tNode = d_nodeArray[tid];
			tNode.mortonCode = morton;

			morton::morton3D_32_decode(morton, x, y, z);
			tNode.origin = gridOrigin + width * Eigen::Vector3d((double)x, (double)y, (double)z);
			tNode.width = width;
		}
	}
}

void SparseVoxelOctree::createOctree(const size_t& nModelTris, const vector<Triangle<V3d>>& modelTris, const AABox<Eigen::Vector3d>& modelBBox, const std::string& base_filename)
{
	assert(surfaceVoxelGridSize.x() >= 1 && surfaceVoxelGridSize.y() >= 1 && surfaceVoxelGridSize.z() >= 1);
	size_t gridCNodeSize = (size_t)morton::mortonEncode_LUT((uint16_t)(surfaceVoxelGridSize.x() - 1), (uint16_t)(surfaceVoxelGridSize.y() - 1), (uint16_t)(surfaceVoxelGridSize.z() - 1)) + 1;
	size_t gridTreeNodeSize = gridCNodeSize % 8 ? gridCNodeSize + 8 - (gridCNodeSize % 8) : gridCNodeSize;
	Eigen::Vector3d unitVoxelSize = Eigen::Vector3d(modelBBox.boxWidth.x() / surfaceVoxelGridSize.x(),
		modelBBox.boxWidth.y() / surfaceVoxelGridSize.y(),
		modelBBox.boxWidth.z() / surfaceVoxelGridSize.z());
	double unitNodeWidth = unitVoxelSize.x();

	Eigen::Vector3i* d_surfaceVoxelGridSize;
	CUDA_CHECK(cudaMalloc((void**)&d_surfaceVoxelGridSize, sizeof(Eigen::Vector3i)));
	CUDA_CHECK(cudaMemcpy(d_surfaceVoxelGridSize, &surfaceVoxelGridSize, sizeof(Eigen::Vector3i), cudaMemcpyHostToDevice));
	Eigen::Vector3d* d_gridOrigin;
	CUDA_CHECK(cudaMalloc((void**)&d_gridOrigin, sizeof(Eigen::Vector3d)));
	CUDA_CHECK(cudaMemcpy(d_gridOrigin, &modelBBox.boxOrigin, sizeof(Eigen::Vector3d), cudaMemcpyHostToDevice));
	Eigen::Vector3d* d_unitVoxelSize;
	CUDA_CHECK(cudaMalloc((void**)&d_unitVoxelSize, sizeof(Eigen::Vector3d)));
	CUDA_CHECK(cudaMemcpy(d_unitVoxelSize, &unitVoxelSize, sizeof(Eigen::Vector3d), cudaMemcpyHostToDevice));
	double* d_unitNodeWidth;
	CUDA_CHECK(cudaMalloc((void**)&d_unitNodeWidth, sizeof(double)));
	CUDA_CHECK(cudaMemcpy(d_unitNodeWidth, &unitNodeWidth, sizeof(double), cudaMemcpyHostToDevice));

	thrust::device_vector<uint32_t> d_CNodeMortonArray(gridCNodeSize, 0);
	thrust::device_vector<bool> d_isValidCNodeArray;
	thrust::device_vector<size_t> d_esumCNodesArray; // exclusive scan
	thrust::device_vector<uint32_t> d_pactCNodeArray;
	thrust::device_vector<short int> d_numTreeNodesArray; // 节点数量记录数组
	thrust::device_vector<size_t> d_sumTreeNodesArray; // inlusive scan
	thrust::device_vector<size_t> d_esumTreeNodesArray; // 存储每一层节点数量的exclusive scan数组
	thrust::device_vector<uint32_t> d_begMortonArray;
	thrust::device_vector<SVONode> d_nodeArray; // 存储某一层的节点数组
	thrust::device_vector<SVONode> d_SVONodeArray; // save all sparse octree nodes

	// mesh voxelize
	resizeThrust(d_CNodeMortonArray, gridCNodeSize, (uint32_t)0);
	meshVoxelize(nModelTris, modelTris, d_surfaceVoxelGridSize, d_unitVoxelSize, d_gridOrigin, d_CNodeMortonArray);

	// create octree
	while (true)
	{
		// compute the number of 'coarse nodes'(eg: voxels)
		resizeThrust(d_isValidCNodeArray, gridCNodeSize);
		resizeThrust(d_esumCNodesArray, gridCNodeSize);
		thrust::transform(d_CNodeMortonArray.begin(), d_CNodeMortonArray.end(), d_isValidCNodeArray.begin(), scanMortonFlag<uint32_t>());
		thrust::exclusive_scan(d_isValidCNodeArray.begin(), d_isValidCNodeArray.end(), d_esumCNodesArray.begin(), 0); // 必须加init
		size_t numCNodes = *(d_esumCNodesArray.rbegin()) + *(d_isValidCNodeArray.rbegin());
		if (!numCNodes) { printf("\n-- Sparse Voxel Octree depth: %d\n", treeDepth); break; }

		treeDepth++;

		// compact coarse node array
		d_pactCNodeArray.clear(); resizeThrust(d_pactCNodeArray, numCNodes);
		getOccupancyMaxPotentialBlockSize(gridCNodeSize, minGridSize, blockSize, gridSize, compactArray, 0, 0);
		compactArray << <gridSize, blockSize >> > (gridCNodeSize, d_isValidCNodeArray.data().get(),
			d_CNodeMortonArray.data().get(), d_esumCNodesArray.data().get(), d_pactCNodeArray.data().get());
		getLastCudaError("Kernel 'compactArray' launch failed!\n");
		vector<uint32_t> h_pactCNodeArray(numCNodes, 0);
		CUDA_CHECK(cudaMemcpy(h_pactCNodeArray.data(), d_pactCNodeArray.data().get(), sizeof(uint32_t) * numCNodes, cudaMemcpyDeviceToHost));

		if (treeDepth == 1)
		{
			numVoxels = numCNodes;
#ifndef NDEBUG
			// 验证体素
			vector<uint32_t> voxelArray;
			voxelArray.resize(numCNodes);
			CUDA_CHECK(cudaMemcpy(voxelArray.data(), d_pactCNodeArray.data().get(), sizeof(uint32_t) * numCNodes, cudaMemcpyDeviceToHost));
			saveVoxel(modelBBox, voxelArray, base_filename, unitNodeWidth);
#endif // !NDEBUG
		}

		// compute the number of (real)octree nodes by coarse node array and set parent's morton code to 'd_CNodeMortonArray'
		size_t numNodes = 1;
		if (numCNodes > 1)
		{
			resizeThrust(d_numTreeNodesArray, numCNodes, (short int)0);
			d_CNodeMortonArray.clear(); resizeThrust(d_CNodeMortonArray, gridTreeNodeSize, (uint32_t)0); // 此时用于记录父节点层的coarse node
			getOccupancyMaxPotentialBlockSize(numCNodes, minGridSize, blockSize, gridSize, cpNumNodes, 0, 0);
			const uint32_t firstMortonCode = getParentMorton(d_pactCNodeArray[0]);
			d_CNodeMortonArray[firstMortonCode] = firstMortonCode | E_MORTON_32_FLAG;
			cpNumNodes << <gridSize, blockSize >> > (numCNodes, d_pactCNodeArray.data().get(), d_numTreeNodesArray.data().get(), d_CNodeMortonArray.data().get());
			getLastCudaError("Kernel 'cpNumNodes' launch failed!\n");
			resizeThrust(d_sumTreeNodesArray, numCNodes, (size_t)0); // inlusive scan
			thrust::inclusive_scan(d_numTreeNodesArray.begin(), d_numTreeNodesArray.end(), d_sumTreeNodesArray.begin());

			numNodes = *(d_sumTreeNodesArray.rbegin()) + 8;
		}
		depthNumNodes.emplace_back(numNodes);

		// set octree node array
		d_nodeArray.clear(); resizeThrust(d_nodeArray, numNodes, SVONode());
		d_begMortonArray.clear(); resizeThrust(d_begMortonArray, numCNodes);
		if (treeDepth == 1)
		{
			getOccupancyMaxPotentialBlockSize(numCNodes, minGridSize, blockSize, gridSize, createNode_1);
			createNode_1 << <gridSize, blockSize >> > (numCNodes, d_sumTreeNodesArray.data().get(),
				d_pactCNodeArray.data().get(), d_gridOrigin, d_unitNodeWidth, d_begMortonArray.data().get(), d_nodeArray.data().get());
			getLastCudaError("Kernel 'createNode_1' launch failed!\n");

			d_esumTreeNodesArray.push_back(0);

			numFineNodes = numNodes;
		}
		else
		{
			getOccupancyMaxPotentialBlockSize(numCNodes, minGridSize, blockSize, gridSize, createNode_2);
			createNode_2 << <gridSize, blockSize >> > (numCNodes, *(d_esumTreeNodesArray.rbegin() + 1), *(d_esumTreeNodesArray.rbegin()),
				d_sumTreeNodesArray.data().get(), d_pactCNodeArray.data().get(), d_gridOrigin, d_unitNodeWidth, d_begMortonArray.data().get(),
				d_nodeArray.data().get(), (d_SVONodeArray.data() + (*(d_esumTreeNodesArray.rbegin() + 1))).get());
			getLastCudaError("Kernel 'createNode_2' launch failed!\n");
		}
		auto newEndOfBegMorton = thrust::unique(d_begMortonArray.begin(), d_begMortonArray.end());
		resizeThrust(d_begMortonArray, newEndOfBegMorton - d_begMortonArray.begin());

		blockSize = 256; gridSize = (numNodes + blockSize - 1) / blockSize;
		createRemainNode << <gridSize, blockSize, sizeof(uint32_t)* blockSize / 8 >> > (numNodes, d_gridOrigin, d_unitNodeWidth,
			d_begMortonArray.data().get(), d_nodeArray.data().get());
		getLastCudaError("Kernel 'createRemainNode' launch failed!\n");

		d_SVONodeArray.insert(d_SVONodeArray.end(), d_nodeArray.begin(), d_nodeArray.end());

		d_esumTreeNodesArray.push_back(numNodes + (*d_esumTreeNodesArray.rbegin()));

		uint32_t numParentCNodes = *thrust::max_element(d_CNodeMortonArray.begin(), d_CNodeMortonArray.end());
		bool isValidMorton = (numParentCNodes >> 31) & 1;
		// '+ isValidMorton' to prevent '(numParentNodes & D_MORTON_32_FLAG) = 0'同时正好可以让最后的大小能存储到最大的莫顿码
		numParentCNodes = (numParentCNodes & D_MORTON_32_FLAG) + isValidMorton;
		if (numParentCNodes != 0)
		{
			resizeThrust(d_CNodeMortonArray, numParentCNodes);
			unitNodeWidth *= 2.0; CUDA_CHECK(cudaMemcpy(d_unitNodeWidth, &unitNodeWidth, sizeof(double), cudaMemcpyHostToDevice));
			gridCNodeSize = numParentCNodes; gridTreeNodeSize = gridCNodeSize % 8 ? gridCNodeSize + 8 - (gridCNodeSize % 8) : gridCNodeSize;
			if (numNodes / 8 == 0) { printf("\n-- Sparse Voxel Octree depth: %d\n", treeDepth); break; }
		}
		else { printf("\n-- Sparse Voxel Octree depth: %d\n", treeDepth); break; }
	}

	// copy to host
	numTreeNodes = d_esumTreeNodesArray[treeDepth];
	svoNodeArray.resize(numTreeNodes);
	auto freeResOfCreateTree = [&]()
	{
		cleanupThrust(d_CNodeMortonArray);
		cleanupThrust(d_isValidCNodeArray);
		cleanupThrust(d_esumCNodesArray);
		cleanupThrust(d_pactCNodeArray);
		cleanupThrust(d_numTreeNodesArray);
		cleanupThrust(d_sumTreeNodesArray);
		cleanupThrust(d_nodeArray);

		CUDA_CHECK(cudaFree(d_surfaceVoxelGridSize));
		CUDA_CHECK(cudaFree(d_gridOrigin));
		CUDA_CHECK(cudaFree(d_unitNodeWidth));
		CUDA_CHECK(cudaFree(d_unitVoxelSize));
	};
	freeResOfCreateTree();

	constructNodeAtrributes(d_esumTreeNodesArray, d_SVONodeArray);
	CUDA_CHECK(cudaMemcpy(svoNodeArray.data(), d_SVONodeArray.data().get(), sizeof(SVONode) * numTreeNodes, cudaMemcpyDeviceToHost));
	cleanupThrust(d_numTreeNodesArray);
	cleanupThrust(d_SVONodeArray);
}

namespace {
	__device__ size_t d_topNodeIdx;
}
template<bool topFlag>
__global__ void findNeighbors(const size_t nNodes,
	const size_t preESumTreeNodes,
	SVONode* d_nodeArray)
{
	if (topFlag)
	{
		d_nodeArray[0].neighbors[13] = d_topNodeIdx;
	}
	else
	{
		size_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
		size_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;

		if (tid_x < nNodes && tid_y < 27)
		{
			SVONode& t = d_nodeArray[preESumTreeNodes + tid_x];
			const SVONode& p = d_nodeArray[t.parent];
			const uint8_t key = (t.mortonCode) & LOWER_3BIT_MASK;
			const unsigned int p_neighborIdx = p.neighbors[morton::neighbor_LUTparent[key][tid_y]];
			if (p_neighborIdx != UINT32_MAX)
			{
				const SVONode& h = d_nodeArray[p_neighborIdx];
				t.neighbors[tid_y] = h.childs[morton::neighbor_LUTchild[key][tid_y]];
			}
			else t.neighbors[tid_y] = UINT32_MAX;
		}
	}

}

void SparseVoxelOctree::constructNodeNeighbors(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
	thrust::device_vector<SVONode>& d_SVONodeArray)
{
	dim3 gridSize, blockSize;
	blockSize.x = 32, blockSize.y = 32;
	gridSize.y = 1;
	// find neighbors(up to bottom)
	if (treeDepth >= 1)
	{
		const size_t idx = d_SVONodeArray.size() - 1;
		CUDA_CHECK(cudaMemcpyToSymbol(d_topNodeIdx, &idx, sizeof(size_t)));
		findNeighbors<true> << <1, 1 >> > (1, 0, (d_SVONodeArray.data() + idx).get());
		for (int i = treeDepth - 2; i >= 0; --i)
		{
			const size_t nNodes = depthNumNodes[i];
			gridSize.x = (nNodes + blockSize.x - 1) / blockSize.x;
			findNeighbors<false> << <gridSize, blockSize >> > (nNodes, d_esumTreeNodesArray[i], d_SVONodeArray.data().get());
		}
	}
}

__global__ void determineNodeVertex(const size_t nNodes,
	const size_t nodeOffset,
	const SVONode* d_nodeArray,
	node_vertex_type* d_nodeVertArray)
{
	size_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid_x < nNodes)
	{
		size_t nodeIdx = nodeOffset + tid_x;
		uint16_t x, y, z;
		const Eigen::Vector3d& origin = d_nodeArray[nodeIdx].origin;
		const double& width = d_nodeArray[nodeIdx].width;

		/*Eigen::Vector3d verts[8] =
		{
			origin,
			Eigen::Vector3d(origin.x() + width, origin.y(), origin.z()),
			Eigen::Vector3d(origin.x(), origin.y() + width, origin.z()),
			Eigen::Vector3d(origin.x() + width, origin.y() + width, origin.z()),

			Eigen::Vector3d(origin.x(), origin.y(), origin.z() + width),
			Eigen::Vector3d(origin.x() + width, origin.y(), origin.z() + width),
			Eigen::Vector3d(origin.x(), origin.y() + width, origin.z() + width),
			Eigen::Vector3d(origin.x() + width, origin.y() + width, origin.z() + width),
		};

		for (int i = 0; i < 8; ++i)
		{
			size_t idx = tid_x;
			for (int j = 0; j < 8; ++j)
			{
				if (d_nodeArray[nodeIdx].neighbors[morton::d_vertSharedLUT[i * 8 + j]] < idx) idx = d_nodeArray[nodeIdx].neighbors[morton::d_vertSharedLUT[i * 8 + j]];
			}
			d_nodeVertArray[tid_x * 8 + i] = thrust::make_pair<Eigen::Vector3d, uint32_t>(verts[i], idx);
		}*/

#pragma unroll
		for (int i = 0; i < 8; ++i)
		{
			morton::morton3D_32_decode(i, x, y, z);
			Eigen::Vector3d corner = origin + width * Eigen::Vector3d(x, y, z);
			size_t idx = nodeIdx;
#pragma unroll
			for (int j = 0; j < 8; ++j)
				if (d_nodeArray[nodeIdx].neighbors[morton::d_vertSharedLUT[i * 8 + j]] < idx) idx = d_nodeArray[nodeIdx].neighbors[morton::d_vertSharedLUT[i * 8 + j]];

			d_nodeVertArray[tid_x * 8 + i] = thrust::make_pair(corner, idx);
		}
	}
}

__global__ void determineNodeEdge(const size_t nNodes,
	const SVONode* d_nodeArray,
	node_edge_type* d_nodeEdgeArray)
{
	size_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid_x < nNodes)
	{
		Eigen::Vector3d origin = d_nodeArray[tid_x].origin;
		double width = d_nodeArray[tid_x].width;

		// 0-2 2-3 1-3 0-1; 4-6 6-7 5-7 4-5; 0-4 2-6 3-7 1-5;
		thrust_edge_type edges[12] =
		{
			thrust::make_pair(origin, origin + Eigen::Vector3d(0, width, 0)),
			thrust::make_pair(origin + Eigen::Vector3d(0, width, 0), origin + Eigen::Vector3d(width, width, 0)),
			thrust::make_pair(origin + Eigen::Vector3d(width, 0, 0), origin + Eigen::Vector3d(width, width, 0)),
			thrust::make_pair(origin,origin + Eigen::Vector3d(width, 0, 0)),

			thrust::make_pair(origin + Eigen::Vector3d(0, 0, width), origin + Eigen::Vector3d(0, width, width)),
			thrust::make_pair(origin + Eigen::Vector3d(0, width, width), origin + Eigen::Vector3d(width, width, width)),
			thrust::make_pair(origin + Eigen::Vector3d(width, 0, width),origin + Eigen::Vector3d(width, width, width)),
			thrust::make_pair(origin + Eigen::Vector3d(0, 0, width), origin + Eigen::Vector3d(width, 0, width)),

			thrust::make_pair(origin, origin + Eigen::Vector3d(0, 0, width)),
			thrust::make_pair(origin + Eigen::Vector3d(0, width, 0), origin + Eigen::Vector3d(0, width, width)),
			thrust::make_pair(origin + Eigen::Vector3d(width, width, 0), origin + Eigen::Vector3d(width, width, width)),
			thrust::make_pair(origin + Eigen::Vector3d(width, 0, 0), origin + Eigen::Vector3d(width, 0, width)),
		};

#pragma unroll
		for (int i = 0; i < 12; ++i)
		{
			thrust_edge_type edge = edges[i];
			size_t idx = tid_x;

#pragma unroll
			for (int j = 0; j < 4; ++j)
				if (d_nodeArray[tid_x].neighbors[morton::d_edgeSharedLUT[i * 4 + j]] < idx) idx = d_nodeArray[tid_x].neighbors[morton::d_edgeSharedLUT[i * 4 + j]];

			d_nodeEdgeArray[tid_x * 12 + i] = thrust::make_pair(edge, idx);
		}
	}
}

template <typename T>
struct lessPoint {
	__host__ __device__ int operator()(const T& a, const T& b) const {
		for (size_t i = 0; i < a.size(); ++i) {
			if (fabs(a[i] - b[i]) < 1e-9) continue;

			if (a[i] < b[i]) return 1;
			else if (a[i] > b[i]) return -1;
		}
		return 0;
	}
};

struct sortVert {
	__host__ __device__ bool operator()(const node_vertex_type& a, const node_vertex_type& b) {
		int _t = lessPoint<V3d>{}(a.first, b.first);
		if (_t == 0) return a.second < b.second;
		else if (_t == 1) return true;
		else return false;
	}
};

struct sortEdge {
	__host__ __device__ bool operator()(node_edge_type& a, node_edge_type& b) {
		int _t_0 = lessPoint<V3d>{}(a.first.first, b.first.first);
		if (_t_0 == 0)
		{
			int _t_1 = lessPoint<V3d>{}(a.first.second, b.first.second);
			if (_t_1 == 0) return a.second < b.second;
			else if (_t_1 == 1) return true;
			else return false;
		}
		else if (_t_0 == 1) return true;
		else return false;
	}
};

struct uniqueVert {
	__host__ __device__ bool operator()(const node_vertex_type& a, const node_vertex_type& b) {
		return (a.first).isApprox(b.first, 1e-9);
	}
};

struct uniqueEdge {
	__host__ __device__
		bool operator()(const node_edge_type& a, const node_edge_type& b) {
		return ((a.first.first.isApprox(b.first.first)) && (a.first.second.isApprox(b.first.second))) ||
			((a.first.first.isApprox(b.first.second)) && (a.first.second.isApprox(b.first.first)));
	}
};

#define MAX_STREAM 16
void SparseVoxelOctree::constructNodeVertexAndEdge(const thrust::device_vector<size_t>& d_esumTreeNodesArray, thrust::device_vector<SVONode>& d_SVONodeArray)
{
	assert(treeDepth + 1 <= MAX_STREAM, "the number of stream is too small!\n");
	cudaStream_t streams[MAX_STREAM];
	for (int i = 0; i < MAX_STREAM; ++i) CUDA_CHECK(cudaStreamCreate(&streams[i]));

	depthNodeVertexArray.resize(treeDepth);
	esumDepthNodeVerts.resize(treeDepth + 1, 0);
	for (int i = 0; i < treeDepth; ++i)
	{
		const size_t numNodes = depthNumNodes[i];
		thrust::device_vector<node_vertex_type> d_nodeVertArray(numNodes * 8);

		getOccupancyMaxPotentialBlockSize(numNodes, minGridSize, blockSize, gridSize, determineNodeVertex, 0, 0);
		determineNodeVertex << <gridSize, blockSize, 0, streams[i] >> > (numNodes, d_esumTreeNodesArray[i], d_SVONodeArray.data().get(), d_nodeVertArray.data().get());
		getLastCudaError("Kernel 'determineNodeVertex' launch failed!\n");

		thrust::sort(thrust::cuda::par.on(streams[i]), d_nodeVertArray.begin(), d_nodeVertArray.end(), sortVert());
		auto vertNewEnd = thrust::unique(thrust::cuda::par.on(streams[i]), d_nodeVertArray.begin(), d_nodeVertArray.end(), uniqueVert());
		cudaStreamSynchronize(streams[i]);

		size_t cur_numNodeVerts = vertNewEnd - d_nodeVertArray.begin();
		resizeThrust(d_nodeVertArray, cur_numNodeVerts);
		numNodeVerts += cur_numNodeVerts;

		std::vector<node_vertex_type> h_nodeVertArray;
		h_nodeVertArray.resize(cur_numNodeVerts);
		CUDA_CHECK(cudaMemcpy(h_nodeVertArray.data(), d_nodeVertArray.data().get(), sizeof(node_vertex_type) * cur_numNodeVerts, cudaMemcpyDeviceToHost));
		depthNodeVertexArray[i] = h_nodeVertArray;
		esumDepthNodeVerts[i + 1] = esumDepthNodeVerts[i] + cur_numNodeVerts;
	}

	thrust::device_vector <node_edge_type> d_fineNodeEdgeArray(numFineNodes * 12);
	getOccupancyMaxPotentialBlockSize(numFineNodes, minGridSize, blockSize, gridSize, determineNodeEdge, 0, 0);
	determineNodeEdge << <gridSize, blockSize, 0, streams[treeDepth] >> > (numFineNodes, d_SVONodeArray.data().get(), d_fineNodeEdgeArray.data().get());
	getLastCudaError("Kernel 'determineNodeEdge' launch failed!\n");

	thrust::sort(thrust::cuda::par.on(streams[treeDepth]), d_fineNodeEdgeArray.begin(), d_fineNodeEdgeArray.end(), sortEdge());
	auto edgeNewEnd = thrust::unique(thrust::cuda::par.on(streams[treeDepth]), d_fineNodeEdgeArray.begin(), d_fineNodeEdgeArray.end(), uniqueEdge());
	cudaStreamSynchronize(streams[treeDepth]);

	numFineNodeEdges = edgeNewEnd - d_fineNodeEdgeArray.begin();
	resizeThrust(d_fineNodeEdgeArray, numFineNodeEdges);
	fineNodeEdgeArray.resize(numFineNodeEdges);
	CUDA_CHECK(cudaMemcpy(fineNodeEdgeArray.data(), d_fineNodeEdgeArray.data().get(), sizeof(node_edge_type) * numFineNodeEdges, cudaMemcpyDeviceToHost));

	for (int i = 0; i < MAX_STREAM; ++i) CUDA_CHECK(cudaStreamDestroy(streams[i]));
}

void SparseVoxelOctree::constructNodeAtrributes(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
	thrust::device_vector<SVONode>& d_SVONodeArray)
{
	constructNodeNeighbors(d_esumTreeNodesArray, d_SVONodeArray);

	constructNodeVertexAndEdge(d_esumTreeNodesArray, d_SVONodeArray);
}

std::tuple<vector<std::pair<V3d, double>>, vector<size_t>> SparseVoxelOctree::setInDomainPoints(const uint32_t nodeIdx, const int& nodeDepth,
	const vector<size_t>& esumDepthNodeVertexSize, vector<std::map<V3d, size_t>>& nodeVertex2Idx)
{
	int parentDepth = nodeDepth + 1;
	auto parentIdx = svoNodeArray[nodeIdx].parent;
	vector<std::pair<V3d, double>> dm_points;
	vector<size_t> dm_pointsIdx;

	auto getCorners = [&](const SVONode& node, const int& depth)
	{
		const V3d nodeOrigin = node.origin;
		const double nodeWidth = node.width;
		const size_t& esumNodeVerts = esumDepthNodeVertexSize[depth];

		for (int k = 0; k < 8; ++k)
		{
			const int xOffset = k & 1;
			const int yOffset = (k >> 1) & 1;
			const int zOffset = (k >> 2) & 1;

			V3d corner = nodeOrigin + nodeWidth * V3d(xOffset, yOffset, zOffset);

			dm_points.emplace_back(std::make_pair(corner, nodeWidth));
			dm_pointsIdx.emplace_back(esumNodeVerts + nodeVertex2Idx[depth][corner]);
		}
	};

	while (parentIdx != UINT_MAX)
	{
		const auto& svoNode = svoNodeArray[parentIdx];
		getCorners(svoNode, parentDepth);
		parentIdx = svoNode.parent;
		parentDepth++;
	}

	return std::make_tuple(dm_points, dm_pointsIdx);
}

//////////////////////
//  I/O: Save Data  //
//////////////////////
void SparseVoxelOctree::saveSVO(const std::string& filename) const
{
	std::ofstream output(filename.c_str(), std::ios::out);
	assert(output);

#ifndef SILENT
	std::cout << "[I/O] Writing Sparse Voxel Octree data in obj format to file " << std::quoted(filename.c_str()) << std::endl;
	// Write stats
	size_t voxels_seen = 0;
	const size_t write_stats_25 = numTreeNodes / 4.0f;
	fprintf(stdout, "[I/O] Writing to file: 0%%...");
#endif

	size_t faceBegIdx = 0;
	for (const auto& node : svoNodeArray)
	{
#ifndef SILENT			
		voxels_seen++;
		if (voxels_seen == write_stats_25) { fprintf(stdout, "25%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(2)) { fprintf(stdout, "50%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(3)) { fprintf(stdout, "75%%..."); }
#endif
		gvis::writeCube(node.origin, Eigen::Vector3d(node.width, node.width, node.width), output, faceBegIdx);
	}
#ifndef SILENT
	fprintf(stdout, "100%% \n");
#endif

	output.close();
}

void SparseVoxelOctree::saveVoxel(const AABox<Eigen::Vector3d>& modelBBox, const vector<uint32_t>& voxelArray,
	const std::string& base_filename, const double& width) const
{
	std::string filename_output = base_filename + std::string("_voxel.obj");
	std::ofstream output(filename_output.c_str(), std::ios::out);
	assert(output);

#ifndef SILENT
	std::cout << "[I/O] Writing data in obj voxels format to file " << std::quoted(filename_output.c_str()) << std::endl;
	// Write stats
	size_t voxels_seen = 0;
	const size_t write_stats_25 = voxelArray.size() / 4.0f;
	fprintf(stdout, "[I/O] Writing to file: 0%%...");
#endif

	size_t faceBegIdx = 0;
	for (size_t i = 0; i < voxelArray.size(); ++i)
	{
#ifndef SILENT			
		voxels_seen++;
		if (voxels_seen == write_stats_25) { fprintf(stdout, "25%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(2)) { fprintf(stdout, "50%%..."); }
		else if (voxels_seen == write_stats_25 * size_t(3)) { fprintf(stdout, "75%%..."); }
#endif

		const auto& morton = voxelArray[i];

		uint16_t x, y, z;
		morton::morton3D_32_decode((morton & D_MORTON_32_FLAG), x, y, z);
		const Eigen::Vector3d nodeOrigin = modelBBox.boxOrigin + width * Eigen::Vector3d((double)x, (double)y, (double)z);
		gvis::writeCube(nodeOrigin, Eigen::Vector3d(width, width, width), output, faceBegIdx);
	}
#ifndef SILENT
	fprintf(stdout, "100%% \n");
#endif

	output.close();
}
