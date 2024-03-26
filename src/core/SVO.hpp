#pragma once

#include "Config.hpp"
#include "detail/BasicDataType.hpp"
#include "detail/Geometry.hpp"
#include "utils/Common.hpp"
#include <utility>
#include <vector>
#include <map>
#include <limits>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <Eigen/Dense>

NAMESPACE_BEGIN(ITS)
    namespace svo {
        using namespace Eigen;

        struct SVONode {
            uint32_t mortonCode = 0;
            bool isLeaf = true;

            Vector3d origin;
            double width;

            unsigned int parent{UINT_MAX};
            unsigned int childs[8]{UINT_MAX};
            unsigned int neighbors[27]{UINT_MAX};

            CUDA_GENERAL_CALL SVONode() {
                utils::Loop<unsigned int, 8>([&](auto i) { childs[i] = UINT_MAX; });
                utils::Loop<unsigned int, 27>([&](auto i) { neighbors[i] = UINT_MAX; });
            }
        };

        using thrust_edge_type = thrust::pair<Eigen::Vector3d, Eigen::Vector3d>;
        using node_edge_type = thrust::pair<thrust_edge_type, uint32_t>;
        using node_vertex_type = thrust::pair<Eigen::Vector3d, uint32_t>;

        class SparseVoxelOctree {
        public:
            SparseVoxelOctree() : treeDepth(0) {}

            explicit SparseVoxelOctree(Eigen::Vector3i _gridSize) :
                    treeDepth(0), surfaceVoxelGridSize(std::move(_gridSize)) {}

            SparseVoxelOctree(const int &_grid_x, const int &_grid_y, const int &_grid_z) :
                    treeDepth(0), surfaceVoxelGridSize(Eigen::Vector3i(_grid_x, _grid_y, _grid_z)) {}

        public:
            void createOctree(const size_t &nModelTris, const std::vector<Triangle<Vector3d>> &modelTris,
                              const AABox<Eigen::Vector3d> &modelBBox, const std::string &base_filename);

            // save data
            void saveSVO(const std::string &filename) const;

        public:
            void meshVoxelize(const size_t &nModelTris,
                              const std::vector<Triangle<Vector3d>> &modelTris,
                              const Eigen::Vector3i *d_surfaceVoxelGridSize,
                              const Eigen::Vector3d *d_unitVoxelSize,
                              const Eigen::Vector3d *d_gridOrigin,
                              thrust::device_vector<uint32_t> &d_CNodeMortonArray); // construct nodes in `depth - 1`

            void saveVoxel(const AABox<Eigen::Vector3d> &modelBBox, const std::vector<uint32_t> &voxelArray,
                           const std::string &base_filename, const double &width) const;

            void constructNodeNeighbors(const thrust::device_vector<size_t> &d_esumTreeNodesArray,
                                        thrust::device_vector<SVONode> &d_SVONodeArray);

            void constructNodeVertexAndEdge(const thrust::device_vector<size_t> &d_esumTreeNodesArray,
                                            thrust::device_vector<SVONode> &d_SVONodeArray);

            void constructNodeAtrributes(const thrust::device_vector<size_t> &d_esumTreeNodesArray,
                                         thrust::device_vector<SVONode> &d_SVONodeArray);

            /*std::tuple<vector<std::pair<Vector3d, double>>, vector<size_t>> setInDomainPoints(const uint32_t& nodeIdx,
                const int& nodeDepth, vector<std::map<Vector3d, size_t>>& nodeVertex2Idx) const;*/

            std::vector<std::tuple<Vector3d, double, size_t>>
            setInDomainPoints(uint32_t _nodeIdx, int nodeDepth,
                              const std::vector<std::map<Vector3d, size_t>> &depthVert2Idx);

            // multi point query test
            std::vector<std::tuple<Vector3d, double, size_t>>
            mq_setInDomainPoints(uint32_t _morton, const Vector3d &modelOrigin,
                                 double _searchNodeWidth,
                                 int _searchDepth,
                                 const std::vector<std::map<uint32_t, uint32_t>> &depthMorton2Nodes,
                                 const std::vector<std::map<Vector3d, size_t>> &depthVert2Idx) const;

        public:
            int treeDepth = 0;
            size_t numVoxels = 0;
            size_t numFineNodes = 0;
            size_t numTreeNodes = 0;

            Eigen::Vector3i surfaceVoxelGridSize;
            std::vector<size_t> depthNumNodes; // ÿһ��İ˲����ڵ���
            //vector<vector<SVONode>> SVONodes;

            std::vector<std::vector<SVONode>> depthSVONodeArray; // ÿһ��İ˲����ڵ�
            std::vector<SVONode> svoNodeArray; // ���еİ˲����ڵ�

            std::vector<node_vertex_type> nodeVertexArray;
            std::vector<std::vector<node_vertex_type>> depthNodeVertexArray;
            std::vector<node_edge_type> fineNodeEdgeArray;

            size_t numNodeVerts = 0;
            size_t numFineNodeEdges = 0;

            std::vector<size_t> esumDepthNodeVerts;
        };

    } // namespace svo
NAMESPACE_END(ITS)