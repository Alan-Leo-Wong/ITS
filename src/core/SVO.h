#pragma once
#include <vector>
#include <limits>
#include <Eigen/Dense>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
//#include "ThinShells.h"
#include "BasicDataType.h"
#include "utils/Common.hpp"
#include "utils/Geometry.hpp"
#include "utils/cuda\CUDAMacro.h"

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

using thrust_edge_type = thrust::pair<Eigen::Vector3d, Eigen::Vector3d>;
using node_edge_type = thrust::pair<thrust_edge_type, uint32_t>;
using node_vertex_type = thrust::pair<Eigen::Vector3d, uint32_t>;

class ThinShells;
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
	//vector<vector<SVONode>> SVONodes;

	vector<vector<SVONode>> depthSVONodeArray; // 每一层的八叉树节点
	vector<SVONode> svoNodeArray; // 所有的八叉树节点

	vector<node_vertex_type> nodeVertexArray;
	vector<vector<node_vertex_type>> depthNodeVertexArray;
	vector<node_edge_type> fineNodeEdgeArray;

	size_t numNodeVerts = 0;
	size_t numFineNodeEdges = 0;

	vector<size_t> esumDepthNodeVerts;

private:
	void meshVoxelize(const size_t& nModelTris,
		const vector<Triangle<V3d>>& modelTris,
		const Eigen::Vector3i* d_surfaceVoxelGridSize,
		const Eigen::Vector3d* d_unitVoxelSize,
		const Eigen::Vector3d* d_gridOrigin,
		thrust::device_vector<uint32_t>& d_CNodeMortonArray); // construct nodes in `depth - 1`

	void saveVoxel(const AABox<Eigen::Vector3d>& modelBBox, const vector<uint32_t>& voxelArray,
		const std::string& base_filename, const double& width) const;

	void constructNodeNeighbors(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
		thrust::device_vector<SVONode>& d_SVONodeArray);

	void constructNodeVertexAndEdge(const thrust::device_vector<size_t>& d_esumTreeNodesArray, thrust::device_vector<SVONode>& d_SVONodeArray);

	void constructNodeAtrributes(const thrust::device_vector<size_t>& d_esumTreeNodesArray,
		thrust::device_vector<SVONode>& d_SVONodeArray);

	/*std::tuple<vector<std::pair<V3d, double>>, vector<size_t>> setInDomainPoints(const uint32_t& nodeIdx, 
		const int& nodeDepth, vector<std::map<V3d, size_t>>& nodeVertex2Idx) const;*/

	std::vector<std::tuple<V3d, double, size_t>> setInDomainPoints(const uint32_t& _nodeIdx, const int& nodeDepth, 
		vector<std::map<V3d, size_t>>& depthVert2Idx);

	// multi point query test
	std::vector<std::tuple<V3d, double, size_t>> mq_setInDomainPoints(const uint32_t& _morton, const V3d& modelOrigin,
		const double& _searchNodeWidth, const int& _searchDepth, vector<std::map<uint32_t, uint32_t>>& depthMorton2Nodes, 
		vector<std::map<V3d, size_t>>& depthVert2Idx) const;

public:
	SparseVoxelOctree() : treeDepth(0) {}

	SparseVoxelOctree(const Eigen::Vector3i& _gridSize) :
		treeDepth(0), surfaceVoxelGridSize(_gridSize) {}

	SparseVoxelOctree(const int& _grid_x, const int& _grid_y, const int& _grid_z) :
		treeDepth(0), surfaceVoxelGridSize(Eigen::Vector3i(_grid_x, _grid_y, _grid_z)) {}

public:
	void createOctree(const size_t& nModelTris, const vector<Triangle<V3d>>& modelTris, const AABox<Eigen::Vector3d>& modelBBox, const std::string& base_filename);

public:
	// save data
	void saveSVO(const std::string& filename) const;
};