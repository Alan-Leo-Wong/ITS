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

            explicit SparseVoxelOctree(Eigen::Vector3i _gridSize, const AABox<Eigen::Vector3d> &_modelBBox) :
                    treeDepth(0), surfaceVoxelGridSize(std::move(_gridSize)), modelBoundingBox(_modelBBox) {}

            explicit SparseVoxelOctree(int _grid, const AABox<Eigen::Vector3d> &_modelBBox) :
                    treeDepth(0), surfaceVoxelGridSize(Eigen::Vector3i(_grid, _grid, _grid)),
                    modelBoundingBox(_modelBBox) {}

        private:
            /**
             * Voxelize the input mesh.
             * @param nModelTris
             * @param modelTris
             * @param d_surfaceVoxelGridSize
             * @param d_unitVoxelSize
             * @param d_gridOrigin
             * @param d_CNodeMortonArray
             */
            void meshVoxelize(const size_t &nModelTris,
                              const std::vector<Triangle<Vector3d>> &modelTris,
                              const Eigen::Vector3i *d_surfaceVoxelGridSize,
                              const Eigen::Vector3d *d_unitVoxelSize,
                              const Eigen::Vector3d *d_gridOrigin,
                              thrust::device_vector<uint32_t> &d_CNodeMortonArray); // construct nodes in `depth - 1`

            /**
             * Construct each node's neighbors.
             * @param d_esumTreeNodesArray
             * @param d_SVONodeArray
             */
            void constructNodeNeighbors(const thrust::device_vector<size_t> &d_esumTreeNodesArray,
                                        thrust::device_vector<SVONode> &d_SVONodeArray);

            /**
             * Construct nodes' edge and vertex array.
             * @param d_esumTreeNodesArray
             * @param d_SVONodeArray
             */
            void constructNodeVertexAndEdge(const thrust::device_vector<size_t> &d_esumTreeNodesArray,
                                            thrust::device_vector<SVONode> &d_SVONodeArray);

            /**
             * Construct each node's neighbors, and nodes' edge and vertex array.
             * @param d_esumTreeNodesArray
             * @param d_SVONodeArray
             */
            void constructNodeAtrributes(const thrust::device_vector<size_t> &d_esumTreeNodesArray,
                                         thrust::device_vector<SVONode> &d_SVONodeArray);

        public:
            /**
             * [API]: Create the sparse voxel octree.
             * @param nModelTris the number of triangles of the input mesh
             * @param modelTris triangles data of the input mesh
             * @param modelBBox input mesh's bounding-box
             */
            void createOctree(size_t nModelTris, const std::vector<Triangle<Vector3d>> &modelTris);

        public:
            /*std::tuple<vector<std::pair<Vector3d, double>>, vector<size_t>> setInDomainPoints(const uint32_t& nodeIdx,
                const int& nodeDepth, vector<std::map<Vector3d, size_t>>& nodeVertex2Idx) const;*/

            /**
             * [API]: Select node corners that influence the specific node.
             * @param _nodeIdx query node
             * @param nodeDepth initial search depth
             * @param depthVert2Idx mapping node vertex at each depth to its global index
             * @return influencing node corners, node width, global index
             */
            std::vector<std::tuple<Vector3d, double, size_t>>
            setInDomainPoints(uint32_t _nodeIdx, int nodeDepth,
                              const std::vector<std::map<Vector3d, size_t>> &depthVert2Idx);

            /**
             * [API]: Select node corners that influence the point.
             * It's used for any point query.
             * @param _morton a point's morton code
             * @param modelOrigin
             * @param _searchNodeWidth
             * @param _searchDepth
             * @param depthMorton2Nodes
             * @param depthVert2Idx
             * @return
             */
            std::vector<std::tuple<Vector3d, double, size_t>>
            mq_setInDomainPoints(uint32_t _morton, const Vector3d &modelOrigin,
                                 double _searchNodeWidth,
                                 int _searchDepth,
                                 const std::vector<std::map<uint32_t, uint32_t>> &depthMorton2Nodes,
                                 const std::vector<std::map<Vector3d, size_t>> &depthVert2Idx) const;

        public:
            /**
             * [API]: Save the voxelized mesh.
             * @param modelBBox
             * @param voxelArray
             * @param base_filename
             * @param width
             */
            void saveVoxel(const AABox<Eigen::Vector3d> &modelBBox, const std::vector<uint32_t> &voxelArray,
                           const std::string &base_filename, double width) const;

            /**
             * [API]: Save the sparse voxel octree depth by depth.
             * @param filename
             */
            void saveSVO(const std::string &filename) const;

        public:
            AABox<Eigen::Vector3d> modelBoundingBox;

            int treeDepth = 0;
            size_t numVoxels = 0; // the number of voxels
            size_t numFineNodes = 0; // the number of depth-0 nodes
            size_t numTreeNodes = 0; // the number of svo nodes

        public:
            size_t numNodeVerts = 0; // the number of svo node vertices
            size_t numFineNodeEdges = 0; // the number of depth-0 node edges

        public:
            Eigen::Vector3i surfaceVoxelGridSize;
            std::vector<size_t> depthNumNodes;
            //vector<vector<SVONode>> SVONodes;

            std::vector<std::vector<SVONode>> depthSVONodeArray;
            std::vector<SVONode> svoNodeArray;

            std::vector<node_vertex_type> nodeVertexArray;
            std::vector<std::vector<node_vertex_type>> depthNodeVertexArray;
            std::vector<node_edge_type> fineNodeEdgeArray;

            std::vector<size_t> esumDepthNodeVerts;
        };

    } // namespace svo
NAMESPACE_END(ITS)