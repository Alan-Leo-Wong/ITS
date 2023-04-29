#pragma once
#include <vector>
#include <limits>
#include <Eigen\Dense>
#include <thrust\pair.h>
#include <thrust\device_vector.h>
#include "ThinShells.h"
#include "utils\Common.hpp"
#include "utils\Geometry.hpp"
#include "utils\cuda\CUDAMacro.h"

using std::vector;

struct SVONode
{
	uint32_t mortonCode = 0;
	bool isLeaf = true;

	Eigen::Vector3d origin;
	double width;

	unsigned int parent{ UINT_MAX };
	unsigned int childs[8]{ UINT_MAX };
	unsigned int neighbors[27]{ UINT_MAX };

	_CUDA_GENERAL_CALL_ SVONode()
	{
		detail::Loop<unsigned int, 8>([&](auto i) {childs[i] = UINT_MAX; });
		detail::Loop<unsigned int, 27>([&](auto i) {neighbors[i] = UINT_MAX; });
	}
};

_CUDA_GENERAL_CALL_ uint32_t getParentMorton(const uint32_t morton)
{
	return (((morton >> 3) & 0xfffffff));
}

_CUDA_GENERAL_CALL_ bool isSameParent(const uint32_t morton_1, const uint32_t morton_2)
{
	return getParentMorton(morton_1) == getParentMorton(morton_2);
}

using thrust_edge_type = thrust::pair<Eigen::Vector3d, Eigen::Vector3d>;
using node_edge_type = thrust::pair<thrust_edge_type, uint32_t>;
using node_vertex_type = thrust::pair<Eigen::Vector3d, uint32_t>;
class SparseVoxelOctree
{
friend class ThinShells;
private:
	int treeDepth = 0;
	size_t numVoxels = 0;
	size_t numFineNodes = 0;
	size_t numTreeNodes = 0;

	Eigen::Vector3i surfaceVoxelGridSize;
	vector<size_t> depthNumNodes; // 每一层的八叉树节点数
	vector<vector<SVONode>> SVONodes;

	vector<SVONode> svoNodeArray;

	vector<node_vertex_type> nodeVertexArray;
	vector<node_edge_type> fineNodeEdgeArray;

private:
	void meshVoxelize(const size_t& nModelTris,
		const Eigen::Vector3i* d_surfaceVoxelGridSize,
		const Eigen::Vector3d* d_unitVoxelSize,
		const Eigen::Vector3d* d_gridOrigin,
		thrust::device_vector<uint32_t>& d_CNodeMortonArray); // construct nodes in `depth - 1`

	void constructNodeNeighbors(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
		thrust::device_vector<SVONode>& d_SVONodeArray);

	void constructNodeVertexAndEdge(thrust::device_vector<SVONode>& d_SVONodeArray);

	void constructNodeAtrributes(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
		thrust::device_vector<SVONode>& d_SVONodeArray);

public:
	_CUDA_HOST_CALL_ SparseVoxelOctree() : treeDepth(0) {}

	_CUDA_HOST_CALL_ SparseVoxelOctree(const std::string& filename, const Eigen::Vector3i& _gridSize) :
		treeDepth(0), surfaceVoxelGridSize(_gridSize) {}

	_CUDA_HOST_CALL_ SparseVoxelOctree(const int& _grid_x, const int& _grid_y, const int& _grid_z) :
		treeDepth(0), surfaceVoxelGridSize(Eigen::Vector3i(_grid_x, _grid_y, _grid_z)) {}

public:
	_CUDA_GENERAL_CALL_ void createOctree(const size_t& nModelTris, const AABox<Eigen::Vector3d>& modelBBox, const std::string& base_filename);

public:
	// save data
	_CUDA_GENERAL_CALL_ void saveTree(const std::string& base_filename);

	_CUDA_GENERAL_CALL_ void saveVoxel(const AABox<Eigen::Vector3d>& modelBBox, const vector<uint32_t>& voxelArray,
		const std::string& base_filename, const double& width);
};