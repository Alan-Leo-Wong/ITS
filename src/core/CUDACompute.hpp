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
        //void accIntersection();

        void launch_modelTriAttributeKernel(const size_t &nTriangles,
                                            std::vector<Triangle<Eigen::Vector3d>> &modelTriangleArray);

        //template <typename T>
        void launch_BLASRowSumReduce(const int &rows,
                                     const int &columns,
                                     double *d_matrix,
                                     double *d_row_sums,
                                     const cudaStream_t &stream = nullptr);

        //template <typename T>
        void launch_ThrustRowSumReduce(const int &rows,
                                       const int &columns,
                                       const thrust::device_vector<double> &d_matrix,
                                       thrust::device_vector<double> &row_sums,
                                       const cudaStream_t &stream = nullptr);

        //void cpIntersection();

        /*void cpModelPointsMorton(const Vector3d& modelOrigin, const double& nodeWidth,
            const uint& nModelVerts, const vector<Vector3d> modelVertsArray, vector<uint32_t> vertsMorton);*/

        void cpBSplineVal(const uint &numNodeVerts, const uint &numNodes,
                          const Vector3d &pointData,
                          const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>> &nodeVertexArray,
                          const std::vector<Vector3d> &nodeWidthArray, const VectorXd &lambda, double &bSplinVal,
                          bool useThrust = true);

        void cpBSplineVal(const uint &numPoints, const uint &numNodeVerts, const uint &numNodes,
                          const std::vector<Vector3d> &pointsData,
                          const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>> &nodeVertexArray,
                          const std::vector<Vector3d> &nodeWidthArray, const VectorXd &lambda, VectorXd &bSplinVal,
                          const bool &useThrust = true);

        void cpBSplineVal(const cudaDeviceProp &prop, const uint &numPoints, const uint &numNodeVerts,
                          const thrust::pair<Eigen::Vector3d, uint32_t> *d_nodeVertexArray, const Vector3d *d_nodeWidthArray,
                          const double *d_lambda, const thrust::device_vector<Vector3d> &d_pointsData,
                          thrust::device_vector<double> &d_bSplineVal, const cudaStream_t &stream,
                          bool useThrust = true);

        void cpPointQuery(const uint &numPoints, const uint &numNodeVerts,
                          const uint &numNodes, const Array3d &minRange, const Array3d &maxRange,
                          const std::vector<Vector3d> &pointsData,
                          const std::vector<thrust::pair<Eigen::Vector3d, uint32_t>> &nodeVertexArray,
                          const std::vector<Vector3d> &nodeWidthArray, const VectorXd &lambda, const double &outerVal,
                          VectorXd &bSplinVal, bool useThrust = true);
    }
NAMESPACE_END(ITS)