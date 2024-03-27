#pragma once

#include "Config.hpp"
#include "detail/BasicDataType.hpp"
#include "detail/cuda/CUDACheck.cuh"
#include "detail/cuda/cuBLASCheck.cuh"
#include "detail/Geometry.hpp"
#include <thrust/device_vector.h>

NAMESPACE_BEGIN(ITS)
    namespace cuAcc {
        using namespace Eigen;

        /**
         * Setting triangles' normal and area of a mesh
         * @param nTriangles
         * @param modelTriangleArray
         */
        void launch_modelTriAttributeKernel(size_t nTriangles,
                                            std::vector<Triangle<Vector3d>> &modelTriangleArray);

        /**
         * Matrix sum reduction of each row using cuBLAS.
         * @param rows
         * @param columns
         * @param d_matrix
         * @param d_row_sums
         * @param stream
         */
        void launch_BLASRowSumReduce(int rows,
                                     int columns,
                                     double *d_matrix,
                                     double *d_row_sums,
                                     const cudaStream_t &stream = nullptr);

        /**
         * Matrix sum reduction of each row using Thrust.
         * @param rows
         * @param columns
         * @param d_matrix
         * @param row_sums
         * @param stream
         */
        void launch_ThrustRowSumReduce(int rows,
                                       int columns,
                                       const thrust::device_vector<double> &d_matrix,
                                       thrust::device_vector<double> &row_sums,
                                       const cudaStream_t &stream = nullptr);

        /**
         * Compute B-Spline value on GPU.
         * Arguments are in host and only compute a single point.
         * @param numNodeVerts
         * @param numNodes
         * @param pointData query point
         * @param nodeVertexArray
         * @param nodeWidthArray
         * @param lambda
         * @param bSplinVal result in host
         * @param useThrust
         */
        void cpBSplineVal(uint numNodeVerts, uint numNodes,
                          const Vector3d &pointData,
                          const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>> &nodeVertexArray,
                          const std::vector<Vector3d> &nodeWidthArray, const VectorXd &lambda, double &bSplinVal,
                          bool useThrust = true);

        /**
         * Compute B-Spline value on GPU.
         * Arguments are in host.
         * @param numPoints
         * @param numNodeVerts
         * @param numNodes
         * @param pointsData query points in host
         * @param nodeVertexArray
         * @param nodeWidthArray
         * @param lambda
         * @param bSplinVal result in host
         * @param useThrust
         */
        void cpBSplineVal(uint numPoints, uint numNodeVerts, uint numNodes,
                          const std::vector<Vector3d> &pointsData,
                          const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>> &nodeVertexArray,
                          const std::vector<Vector3d> &nodeWidthArray, const VectorXd &lambda, VectorXd &bSplinVal,
                          bool useThrust = true);

        /**
         * Compute B-Spline value on GPU.
         * Argument are in device.
         * @param prop
         * @param numPoints
         * @param numNodeVerts
         * @param d_nodeVertexArray
         * @param d_nodeWidthArray
         * @param d_lambda
         * @param d_pointsData query points in device
         * @param d_bSplineVal result in device
         * @param stream
         * @param useThrust
         */
        void cpBSplineVal(const cudaDeviceProp &prop, uint numPoints, uint numNodeVerts,
                          const thrust::pair<Eigen::Vector3d, uint32_t> *d_nodeVertexArray, const Vector3d *d_nodeWidthArray,
                          const double *d_lambda, const thrust::device_vector<Vector3d> &d_pointsData,
                          thrust::device_vector<double> &d_bSplineVal, const cudaStream_t &stream,
                          bool useThrust = true);

        /**
         * Compute B-Spline value on GPU.
         * It's used for checking the points are whether in the range of two shells.
         * @param numPoints
         * @param numNodeVerts
         * @param numNodes
         * @param minRange
         * @param maxRange
         * @param pointsData query points in device
         * @param nodeVertexArray
         * @param nodeWidthArray
         * @param lambda
         * @param outerVal
         * @param bSplinVal
         * @param useThrust
         */
        void cpPointQuery(uint numPoints, uint numNodeVerts,
                          uint numNodes, const Array3d &minRange, const Array3d &maxRange,
                          const std::vector<Vector3d> &pointsData,
                          const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>> &nodeVertexArray,
                          const std::vector<Vector3d> &nodeWidthArray, const VectorXd &lambda, const double &outerVal,
                          VectorXd &bSplinVal, bool useThrust = true);
    } // namespace cuAcc
NAMESPACE_END(ITS)