#pragma once

#include "core/SVO.hpp"
#include "detail/cuda/CUDACheck.cuh"
#include "detail/cuda/CUDAMath.hpp"
#include <thrust/host_vector.h>
#include <Eigen/Dense>

NAMESPACE_BEGIN(ITS)

    namespace MCKernel {
        using namespace svo;
        using namespace Eigen;

        template<typename T>
        __host__ __device__

        inline T lerp(T v0, T v1, T t) {
            return fma(t, v1, fma(-t, v0, v0));
        }

        __device__ double3

        vertexLerp(double3
                   p_0,
                   double3 p_1,
                   double sdf_0,
                   double sdf_1,
                   double isoVal
        );

        __device__ uint3

        getVoxelShift(uint
                      index,
                      uint3 d_res
        );

        // 与SDFCompute.cu二选一
        __device__ double
        computeSDF(uint
                   numNodeVerts,
                   const thrust::pair<Eigen::Vector3d, uint32_t> *d_nodeVertexArray,
                   const SVONode *d_svoNodeArray,
                   const double *d_lambda, double3
                   pos);

        /*__global__ void prepareMatrixKernel(const uint nVoxelElemCorners, const uint nAllNodeCorners,
            const uint* d_voxelOffset, const uint3* d_res, double* d_lambda,
            double3* d_origin, double3* d_voxelSize,
            V3d* d_nodeCorners, V3d* d_nodeWidth, double* d_voxelMatrix);*/

        __global__ void prepareVoxelCornerKernel(uint
                                                 nVoxels,
                                                 const uint *voxel_offset,
                                                 const double3 *d_origin,
                                                 const double3 *d_voxelSize,
                                                 const uint3 *d_res,
                                                 V3d
                                                 *d_voxelCornerData);

        __global__ void determineVoxelKernel(uint
                                             nVoxels,
                                             const double *d_isoVal,
                                             cudaTextureObject_t
                                             nVertsTex,
                                             uint *d_nVoxelVerts, uint
                                             *d_voxelCubeIndex,
                                             double *d_voxelSDF, uint
                                             *d_isValidVoxelArray);

        __global__ void determineVoxelKernel_2(uint
                                               nVoxels,
                                               uint nAllNodes,
                                               const double *d_isoVal,
                                               const double3 *d_origin,
                                               const double3 *d_voxelSize,
                                               const uint3 *d_res,
                                               const thrust::pair<Eigen::Vector3d, uint32_t> *nodeVertexArray,
                                               const SVONode *svoNodeArray,
                                               const double *d_lambda,
                                               cudaTextureObject_t
                                               nVertsTex,
                                               uint *d_nVoxelVertsArray,
                                               uint
                                               *d_voxelCubeIndex,
                                               double *d_voxelSDF, uint
                                               *d_isValidVoxelArray);

        __global__ void compactVoxels(uint
                                      nVoxels,
                                      const uint *d_isValidVoxel,
                                      const uint *d_nValidVoxelsScan,
                                      uint
                                      *d_compactedVoxelArray);

        __global__ void voxelToMeshKernel(
                uint
                nValidVoxels,
                uint maxVerts,
                const double *d_isoVal,
                const double3 *d_voxelSize,
                const double3 *d_origin,
                const uint3 *d_res,
                const uint *d_compactedVoxelArray, cudaTextureObject_t
                nVertsTex,
                cudaTextureObject_t triTex, uint
                *d_voxelCubeIndex,
                double *d_voxelSDF, uint
                *d_nVertsScanned,
                double3 *d_triPoints
        );
    } // namespace MCKernel

    namespace MC {
        // ����kernel: prepareMatrix
#ifndef P_NTHREADS_X // node corner
#  define P_NTHREADS_X 64
#endif // !P_NTHREADS_X
#ifndef P_NTHREADS_Y // voxel corner
#  define P_NTHREADS_Y 16
#endif // !P_NTHREADS_Y

        // ����kernel: determine��tomesh
#ifndef V_NTHREADS
#  define V_NTHREADS 256
#endif // !NTHREADS

#ifndef MAX_NUM_STREAMS
#  define MAX_NUM_STREAMS 128 // ���ڴ����з���--voxel�����ֿ���
#endif // !MAX_NUM_STREAMS
    }

    namespace MC {
        // host
        //namespace {
        extern uint numNodeVerts;

        extern uint allTriVertices, nValidVoxels;

        extern double3 *h_triPoints; // output
        //} // namespace

        // device
        //namespace {
        extern double *d_lambda;

        extern uint3 *d_res;
        extern uint3 h_res;

        extern double *d_isoVal;

        extern uint *d_nVoxelVertsArray;
        extern uint *d_nVoxelVertsScan;

        extern uint *d_isValidVoxelArray;
        extern uint *d_nValidVoxelsScan;

        extern double3 *d_gridOrigin;
        extern double3 *d_voxelSize;

        extern thrust::device_vector<double> d_voxelSDF;
        extern thrust::host_vector<double> h_voxelSDF;

        extern uint *d_voxelCubeIndex;

        extern uint *d_compactedVoxelArray;

        extern int *d_triTable;
        extern int *d_nVertsTable;

        // textures containing look-up tables
        extern cudaTextureObject_t triTex;
        extern cudaTextureObject_t nVertsTex;

        extern double3 *d_triPoints; // output
        //} // namespace
    } // namespace MC

    namespace MC {
        using namespace svo;
        using namespace Eigen;

        void d_thrustExclusiveScan(const uint &nElems, uint *input, uint *output);

        void setTextureObject(const uint &srcSizeInBytes, int *srcDev,
                              cudaTextureObject_t *texObj);

        void initCommonResources(const uint &nVoxels,
                                 const uint3 &resolution, const double &isoVal, const double3 &gridOrigin,
                                 const double3 &voxelSize, const uint &maxVerts);

        void freeCommonResources();

        /*void launch_prepareMatrixKernel(const uint& nVoxelElems, const uint& nAllNodes, const uint& voxelOffset,
            const cudaStream_t& stream, double*& d_voxelMatrix);*/

        void launch_computSDFKernel(const uint &nVoxels,
                                    const uint &numNodes, const size_t &_numNodeVerts,
                                    const VectorXd &lambda, const std::vector <V3d> &nodeWidthArray,
                                    const std::vector <thrust::pair<Eigen::Vector3d, uint32_t>> &nodeVertexArray);

        void launch_determineVoxelKernel(const uint &nVoxels, const double &isoVal, const uint &maxVerts);

        void launch_compactVoxelsKernel(const uint &nVoxels);

        void launch_voxelToMeshKernel(const uint &maxVerts, const uint &nVoxels);

        void writeToOBJFile(const std::string &filename);

        void marching_cubes(const std::vector <thrust::pair<Eigen::Vector3d, uint32_t>> &nodeVertexArray,
                            size_t numNodes, const std::vector <V3d> &nodeWidthArray,
                            size_t numSVONodeVerts, const VectorXd &lambda, double3 gridOrigin,
                            double3 gridWidth,
                            uint3 resolution, double isoVal, const std::string &filename);

        void marching_cubes(uint3 resolution,
                            double3 gridOrigin,
                            double3 gridWidth,
                            double isoVal,
                            const thrust::host_vector<double>& gridSDF,
                            const std::string &filename);

    } // namespace MC
NAMESPACE_END(ITS)
